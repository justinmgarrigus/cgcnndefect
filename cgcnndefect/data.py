from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import itertools
import math

import numpy as np
import torch
from pymatgen.core.structure import Structure
from ase.io import read as ase_read
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from .util import ELEM_DICT
from .model_sph_harmonics import get_harmonics_fea


#####################################################
# Cross-validation function
#####################################################
def splitValidation(n, typeVal, parameter, counter):
    """
    Splits dataset indices based on different cross-validation schemes.
    Supports:
      - k-fold (0)
      - bootstrapping (1)
      - leave-p-out (2)
      - leave-one-out (3)
      - monte-carlo (4)
    """
    ret = []
    if typeVal == 0:  # k-fold
        full = list(range(n))
        random.shuffle(full)
        for i in range(parameter):
            test = full[int(i * n / parameter) : int((i + 1) * n / parameter)]
            train = (
                full[0 : int(i * n / parameter)]
                + full[int((i + 1) * n / parameter) : n]
            )
            ret.append({"train": train, "val": test})

    elif typeVal == 1:  # bootstrapping
        for _ in range(parameter):
            valSet = set()
            for _ in range(int(n * 3 / 10)):  # ~30% val
                valSet.add(random.randrange(0, n))
            full = list(range(n))
            train = [idx for idx in full if idx not in valSet]
            ret.append({"train": train, "val": list(valSet)})

    elif typeVal == 2:  # leave-p-out
        combos = list(itertools.combinations(range(n), parameter))
        for indices in combos:
            train = [idx for idx in range(n) if idx not in indices]
            ret.append({"train": train, "val": list(indices)})

    elif typeVal == 3:  # leave-one-out
        for i in range(n):
            ret.append({"train": [x for x in range(n) if x != i], "val": [i]})

    elif typeVal == 4:  # monte carlo
        for _ in range(parameter):
            full = list(range(n))
            random.shuffle(full)
            train = full[: int(n * 0.7)]
            val = full[int(n * 0.7) : int(n * 0.85)]
            test = full[int(n * 0.85) :]
            ret.append({"train": train, "val": val, "test": test})

    return ret[counter]


#####################################################
# get_train_val_test_loader
#####################################################
def get_train_val_test_loader(
    dataset,
    collate_fn=default_collate,
    batch_size=64,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    return_test=False,
    num_workers=1,
    pin_memory=False,
    cross_validation=None,
    cross_param=0,
    counter=0,
    random_seed=None,
    **kwargs
):
    """
    Utility function for dividing a dataset into train/val/test while
    maintaining the Oxygen/Metal ratio AND supporting cross-validation.

    If cross_validation is not None, it uses splitValidation and ignores
    ratio-based logic. Otherwise, it splits Oxygen and Metal indices
    separately, merges them, and respects random_seed for reproducibility.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
    collate_fn: callable
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
    num_workers: int
    pin_memory: bool
    cross_validation: str or None
    cross_param: int
    counter: int
    random_seed: int or None
      if not doing cross-validation, we shuffle with random_seed here.
    kwargs:
      train_size, val_size, test_size if absolute sizes are desired

    Returns
    -------
    train_loader, val_loader, (test_loader if return_test)
    """
    total_size = len(dataset)

    # identify oxygen vs metal
    oxygen_indices = [i for i, data in enumerate(dataset.id_prop_data) if '-O' in data[0]]
    metal_indices  = [i for i, data in enumerate(dataset.id_prop_data) if '-O' not in data[0]]

    if len(oxygen_indices) == 0 or len(metal_indices) == 0:
        raise ValueError("Dataset must contain both Oxygen and Metal entries.")

    # parse ratio logic
    if train_ratio is None:
        assert val_ratio + test_ratio < 1.0, "val_ratio + test_ratio must be < 1.0 if train_ratio is None"
        train_ratio = 1.0 - val_ratio - test_ratio
    else:
        assert train_ratio + val_ratio + test_ratio <= 1.0, "Sum of ratios must be <= 1.0"

    # if user specified absolute sizes
    if 'train_size' in kwargs and kwargs['train_size'] is not None:
        train_size = kwargs['train_size']
    else:
        train_size = int(train_ratio * total_size)
    if 'test_size' in kwargs and kwargs['test_size'] is not None:
        test_size = kwargs['test_size']
    else:
        test_size = int(test_ratio * total_size)
    if 'val_size' in kwargs and kwargs['val_size'] is not None:
        val_size = kwargs['val_size']
    else:
        val_size = int(val_ratio * total_size)

    print(f"Final train/val/test sizes are: {train_size} / {val_size} / {test_size}")

    if cross_validation is not None:
        # cross-validation logic overrides ratio approach
        if random_seed is not None:
            random.seed(random_seed)
        # map from string to numeric typeVal
        cross_num = 0
        if cross_validation in ['k-fold', 'k-fold-cross-validation']:
            cross_num = 0
        elif cross_validation in ['bootstrapping', 'bootstrap']:
            cross_num = 1
        elif cross_validation == 'leave-p-out':
            cross_num = 2
        elif cross_validation == 'leave-one-out':
            cross_num = 3
        elif cross_validation in ['monte-carlo', 'monte-carlo-cross-validation']:
            cross_num = 4

        dictdict = splitValidation(total_size, cross_num, cross_param, counter)
        train_sampler = SubsetRandomSampler(dictdict.get("train"))
        val_sampler   = SubsetRandomSampler(dictdict.get("val"))
        test_sampler  = None
        if "test" in dictdict and return_test:
            test_sampler = SubsetRandomSampler(dictdict.get("test"))

    else:
        # ratio-based approach for O and Metal
        # shuffle each group with random_seed
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(oxygen_indices)
        random.shuffle(metal_indices)

        num_oxy = len(oxygen_indices)
        num_met = len(metal_indices)

        # how many O in train/val/test
        train_oxy_size = int(num_oxy * train_ratio)
        val_oxy_size   = int(num_oxy * val_ratio)
        # test is remainder
        train_met_size = int(num_met * train_ratio)
        val_met_size   = int(num_met * val_ratio)
        # test remainder

        # build splits
        train_oxygen = oxygen_indices[:train_oxy_size]
        val_oxygen   = oxygen_indices[train_oxy_size : train_oxy_size + val_oxy_size]
        test_oxygen  = oxygen_indices[train_oxy_size + val_oxy_size :]

        train_metal  = metal_indices[:train_met_size]
        val_metal    = metal_indices[train_met_size : train_met_size + val_met_size]
        test_metal   = metal_indices[train_met_size + val_met_size :]

        train_indices = train_oxygen + train_metal
        val_indices   = val_oxygen + val_metal
        test_indices  = test_oxygen + test_metal

        # shuffle final merges
        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        # Print the raw counts and ratio
        print(f"Train - Oxygen: {len(train_oxygen)}, Metal: {len(train_metal)}, ratio(O)={len(train_oxygen)/(len(train_oxygen)+len(train_metal)):.2f}")
        print(f"Val   - Oxygen: {len(val_oxygen)},   Metal: {len(val_metal)},   ratio(O)={len(val_oxygen)/(len(val_oxygen)+len(val_metal)):.2f}")
        print(f"Test  - Oxygen: {len(test_oxygen)},  Metal: {len(test_metal)},  ratio(O)={len(test_oxygen)/(len(test_oxygen)+len(test_metal)):.2f}")

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler   = SubsetRandomSampler(val_indices)
        test_sampler  = SubsetRandomSampler(test_indices) if return_test else None

    # build DataLoaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory
    )
    if return_test and test_sampler is not None:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


#####################################################
# Collate function
#####################################################
def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal properties.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_atom_type, batch_nbr_type, batch_nbr_dist, batch_pair_type = [], [], [], []
    batch_nbr_fea_idx_all, batch_gs_fea, batch_gp_fea, batch_gd_fea = [], [], [], []
    batch_global_fea = []
    crystal_atom_idx, batch_target = [], []
    batch_target_Fxyz = []
    batch_cif_ids = []

    base_idx = 0
    for i, (features, target, target_F, cif_id) in enumerate(dataset_list):
        (
            atom_fea_i,
            nbr_fea_i,
            nbr_fea_idx_i,
            atom_type_i,
            nbr_type_i,
            nbr_dist_i,
            pair_type_i,
            global_fea_i,
            nbr_fea_idx_all_i,
            gs_fea_i,
            gp_fea_i,
            gd_fea_i
        ) = features

        n_i = atom_fea_i.shape[0]

        batch_atom_fea.append(atom_fea_i)
        batch_nbr_fea.append(nbr_fea_i)
        batch_nbr_fea_idx.append(nbr_fea_idx_i + base_idx)

        batch_atom_type.append(atom_type_i)
        batch_nbr_type.append(nbr_type_i)
        batch_nbr_dist.append(nbr_dist_i)
        batch_pair_type.append(pair_type_i)

        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_target_Fxyz.append(target_F)
        batch_cif_ids.append(cif_id)

        batch_global_fea.append(global_fea_i)

        batch_nbr_fea_idx_all += [x + base_idx for x in nbr_fea_idx_all_i]
        batch_gs_fea += gs_fea_i
        batch_gp_fea += gp_fea_i
        batch_gd_fea += gd_fea_i

        base_idx += n_i

    try:
        stacked_Fxyz = torch.stack(batch_target_Fxyz, dim=0)
    except:
        stacked_Fxyz = None

    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            torch.cat(batch_atom_type, dim=0),
            torch.cat(batch_nbr_type, dim=0),
            torch.cat(batch_nbr_dist, dim=0),
            torch.cat(batch_pair_type, dim=0),
            torch.Tensor(batch_global_fea),
            torch.cat(batch_nbr_fea_idx_all, dim=0),
            torch.stack(batch_gs_fea, dim=0),
            torch.stack(batch_gp_fea, dim=0),
            torch.stack(batch_gd_fea, dim=0)
        ),
        torch.stack(batch_target, dim=0),
        stacked_Fxyz,
        batch_cif_ids
    )


#####################################################
# GaussianDistance, G2Descriptor, AtomInitializer,
# AtomCustomJSONInitializer, CIFData
#####################################################
class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(- (distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


class G2Descriptor(object):
    """
    Expands interatomic distance by G2 radial basis functions.
    """
    def __init__(self, Rc, etas_offsets_basis=[], large=False):
        if not etas_offsets_basis:
            etas = [0.5, 1.0, 1.5]
            offsets = [1.0, 2.0, 3.0, 4.0, 5.0]
            etas_offsets_basis = list(itertools.product(etas, offsets))
            if large:
                etas_offsets_basis += list(itertools.product([100], [2.0, 2.2, 2.4, 2.6]))
                etas_offsets_basis += list(itertools.product([1000], [1.0, 1.1, 1.3, 1.4, 1.5]))

        self.etas_offsets_basis = etas_offsets_basis
        self.etas = np.array([tup[0] for tup in etas_offsets_basis])
        self.offsets = np.array([tup[1] for tup in etas_offsets_basis])
        self.Rc = Rc

    def expand(self, distances):
        return np.exp(
            - self.etas * ((distances[..., np.newaxis] - self.offsets) ** 2) / (self.Rc ** 2)
        )


class AtomInitializer(object):
    """
    Base class for initializing the vector representation for atoms.
    Use one AtomInitializer per dataset!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {v: k for k, v in self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors from a JSON file, which is a dict
    mapping from element number -> a list representing the feature vector.
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    A dataset for reading CIF files and constructing crystal graphs.
    Contains logic for building neighbor lists, spherical harmonics, etc.
    """
    def __init__(
        self,
        root_dir,
        Fxyz=False,
        all_elems=[0],
        max_num_nbr=12,
        radius=8.0,
        dmin=0,
        step=0.2,
        random_seed=123,
        crys_spec=None,
        atom_spec=None,
        csv_ext='',
        model_type='cgcnn',
        K=4,
        njmax=75,
        init_embed_file='atom_init.json'
    ):
        self.root_dir = root_dir
        self.Fxyz = Fxyz
        self.all_elems = all_elems
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.dmin = dmin
        self.step = step
        self.random_seed = random_seed
        self.crys_spec = crys_spec
        self.atom_spec = atom_spec
        self.csv_ext = csv_ext
        self.model_type = model_type
        self.compute_sph_harm = (self.model_type == 'spooky')
        self.K = K
        self.njmax = njmax
        self.init_embed_file = init_embed_file

        self.reload_data()

    def reload_data(self):
        # basic checks
        assert os.path.exists(self.root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv' + self.csv_ext)
        assert os.path.exists(id_prop_file), f'id_prop.csv{self.csv_ext} not found!'

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Shuffle dataset once, for entire dataset, using random_seed
        random.seed(self.random_seed)
        random.shuffle(self.id_prop_data)

        # load atom initializer
        atom_init_file = os.path.join(self.root_dir, self.init_embed_file)
        assert os.path.exists(atom_init_file), 'atom_init.json not found!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)

        # radial descriptor
        self.gdf = G2Descriptor(Rc=self.radius, large=True)

        # If we need force data
        if self.Fxyz:
            new_data = []
            for row in self.id_prop_data:
                forces_path = os.path.join(self.root_dir, row[0] + "_forces.csv")
                forces = np.loadtxt(forces_path, delimiter=',')
                new_data.append([row[0], row[1], forces])
            self.id_prop_data = new_data

        # If we have global crystal-level features
        if self.crys_spec is not None:
            self.global_fea = []
            for row in self.id_prop_data:
                path = os.path.join(self.root_dir, row[0] + "." + self.crys_spec)
                arr_global = np.loadtxt(path)
                self.global_fea.append(arr_global)

        # If we have local (atom-level) features
        if self.atom_spec is not None:
            self.local_fea = []
            for row in self.id_prop_data:
                path = os.path.join(self.root_dir, row[0] + "." + self.atom_spec)
                arr_local = np.loadtxt(path)
                if len(arr_local.shape) == 1:
                    arr_local = arr_local.reshape(-1, 1)
                self.local_fea.append(arr_local)

        if self.all_elems != [0]:
            pair_elems = list(itertools.combinations_with_replacement(sorted(self.all_elems), 2))
            self.pair_ind = {k: v for v, k in enumerate(pair_elems)}
        else:
            self.pair_ind = {-1: -1}

    def reset_root(self, root_dir):
        self.root_dir = root_dir
        self.reload_data()

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        if self.Fxyz:
            cif_id, target_val, target_Fxyz = self.id_prop_data[idx]
        else:
            cif_id, target_val = self.id_prop_data[idx]
            target_Fxyz = None

        cif_path = os.path.join(self.root_dir, cif_id + '.cif')
        structure = ase_read(cif_path)
        crystal = Structure(
            structure.get_cell(),
            structure.get_chemical_symbols(),
            structure.get_positions(),
            coords_are_cartesian=True
        )

        all_atom_types = [ELEM_DICT[crystal[i].specie.symbol] for i in range(len(crystal))]
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)

        (
            atom_fea,
            nbr_fea,
            nbr_fea_idx,
            atom_type,
            nbr_type,
            nbr_dist,
            pair_type,
            nbr_fea_idx_all,
            gs_fea,
            gp_fea,
            gd_fea
        ) = self.featurize_from_nbr_and_atom_list(all_atom_types, all_nbrs, crystal, cif_id)

        if self.crys_spec is not None:
            global_fea = list(self.global_fea[idx])
        else:
            global_fea = []

        if self.atom_spec is not None:
            local_fea = self.local_fea[idx]
            atom_fea = torch.hstack([atom_fea, torch.Tensor(local_fea)])

        target = torch.Tensor([float(target_val)])
        if target_Fxyz is not None:
            target_Fxyz = torch.Tensor(target_Fxyz)

        if self.Fxyz:
            return (
                (atom_fea, nbr_fea, nbr_fea_idx,
                 atom_type, nbr_type, nbr_dist, pair_type,
                 nbr_fea_idx_all, gs_fea, gp_fea, gd_fea),
                target,
                target_Fxyz,
                cif_id
            )
        else:
            return (
                (atom_fea, nbr_fea, nbr_fea_idx,
                 atom_type, nbr_type, nbr_dist, pair_type,
                 global_fea,
                 nbr_fea_idx_all, gs_fea, gp_fea, gd_fea),
                target,
                None,
                cif_id
            )

    def featurize_from_nbr_and_atom_list(self, all_atom_types, all_nbrs, crystal, cif_id='struct'):
        atom_fea = np.vstack([self.ari.get_atom_fea(num) for num in all_atom_types])
        atom_fea = torch.Tensor(atom_fea)

        all_nbrs_sorted = [sorted(nbr_list, key=lambda x: x[1]) for nbr_list in all_nbrs]

        nbr_fea_idx, nbr_dist = [], []
        nbr_type, pair_type = [], []
        nbr_fea_idx_all = []

        for i, nbr_list in enumerate(all_nbrs_sorted):
            if len(nbr_list) < self.max_num_nbr:
                warnings.warn(
                    f"{cif_id} did not find enough neighbors to build graph. "
                    f"Found {len(nbr_list)}, require {self.max_num_nbr}. "
                    "Consider increasing the radius."
                )
                # pad
                nbr_indices   = [x[2] for x in nbr_list] + [0] * (self.max_num_nbr - len(nbr_list))
                nbr_distances = [x[1] for x in nbr_list] + [self.radius + 1.] * (self.max_num_nbr - len(nbr_list))
                nbr_types     = [all_atom_types[x[2]] for x in nbr_list] + [0] * (self.max_num_nbr - len(nbr_list))
                if self.all_elems != [0]:
                    pair_types = [
                        self.pair_ind[tuple(sorted([all_atom_types[i], all_atom_types[x[2]]]))]
                        for x in nbr_list
                    ] + [-1] * (self.max_num_nbr - len(nbr_list))
                else:
                    pair_types = [-1] * self.max_num_nbr
            else:
                nbr_indices   = [x[2] for x in nbr_list[:self.max_num_nbr]]
                nbr_distances = [x[1] for x in nbr_list[:self.max_num_nbr]]
                nbr_types     = [all_atom_types[x[2]] for x in nbr_list[:self.max_num_nbr]]
                if self.all_elems != [0]:
                    pair_types  = [
                        self.pair_ind[tuple(sorted([all_atom_types[i], all_atom_types[x[2]]]))]
                        for x in nbr_list[:self.max_num_nbr]
                    ]
                else:
                    pair_types  = [-1] * self.max_num_nbr

            nbr_fea_idx.append(nbr_indices)
            nbr_dist.append(nbr_distances)
            nbr_type.append(nbr_types)
            pair_type.append(pair_types)

            if self.compute_sph_harm:
                all_idx = [x[2] for x in nbr_list]
                all_idx = all_idx[:self.njmax]
                all_idx += [0] * (self.njmax - len(all_idx))
                nbr_fea_idx_all.append(torch.LongTensor(all_idx))
            else:
                nbr_fea_idx_all.append(torch.LongTensor([0]))

        if self.compute_sph_harm:
            gs_fea, gp_fea, gd_fea = get_harmonics_fea(crystal, all_nbrs_sorted, self.K, self.radius, self.njmax)
        else:
            gs_fea, gp_fea, gd_fea = [torch.zeros(0)], [torch.zeros(0)], [torch.zeros(0)]

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_dist    = np.array(nbr_dist)
        nbr_fea     = self.gdf.expand(nbr_dist)

        atom_fea    = torch.Tensor(atom_fea)
        nbr_fea     = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        nbr_dist    = torch.Tensor(nbr_dist)

        atom_type   = torch.LongTensor(all_atom_types)
        nbr_type    = torch.LongTensor(nbr_type)
        pair_type   = torch.LongTensor(pair_type)

        return (
            atom_fea,
            nbr_fea,
            nbr_fea_idx,
            atom_type,
            nbr_type,
            nbr_dist,
            pair_type,
            nbr_fea_idx_all,
            gs_fea,
            gp_fea,
            gd_fea
        )


@torch.jit.script
class CIFDataFeaturizer:
    """
    Example TorchScript class if needed.
    """
    def __init__(self):
        self.foo()

    def foo(self):
        return 1
