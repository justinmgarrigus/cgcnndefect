from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import itertools

import numpy as np
import torch
from pymatgen.core.structure import Structure
from ase.io import read as ase_read
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from .util import ELEM_DICT
from .model_sph_harmonics import get_harmonics_fea
import math

def splitValidation(n, typeVal, parameter, counter):
    ret = []
    if typeVal == 0: #k fold cross validation, second parameter is k
        full = list(range(0, n))
        random.shuffle(full)
        for i in range(parameter):
            test = full[(int) (i*n/parameter): (int) ((i+1) * n/parameter)]
            train = full[0: (int) (i*n/parameter)] + full[(int) ((i+1)*n/parameter): n]
            ret.append({"train": train, "val": test})    
            
    elif typeVal == 1: #bootstrapping, parameter is number of times
        for i in range(parameter):
            valSet = set()
            for j in range((int) (n*3/10)): #test will be approx 30% (but will be replacement)
                valSet.add(random.randrange(0, n))
            full = list(range(0, n))
            train = [i for i in full if i not in valSet]
            ret.append({"train": train, "val": list(valSet)})

    elif typeVal == 2: #leave p out, second parameter is ‘p’
        combos = list(itertools.combinations(range(n), parameter))
        for indices in combos:
            train = [i for i in range(n) if i not in indices]
            ret.append({"train": train, "val": list(indices)})
    elif typeVal == 3: #leave one out
        for i in range(n):
            ret.append({"train": [x for x in range(0, n) if x != i], "val": [i]})
    elif typeVal == 4: #monte carlo, second parameter is the number of times we are doing monte carlo
        for i in range(parameter):
            full = list(range(0, n))
            random.shuffle(full)
            train = full[0: (int) (n*7/10)]
            val = full[(int) (n*7/10): (int) (n*8.5/10)]
            test = full[(int) (n*8.5/10): n]
            ret.append({"train": train, "val": val, "test": test})
    return ret[counter]

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, cross_validation = None, cross_param = 0, counter = 0, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    print('num_workers ', num_workers)
    #torch.set_num_threads(1)
    #torch.multiprocessing.set_sharing_strategy('file_system')
    total_size = len(dataset)
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if cross_validation == None:
        if train_ratio is None:
            assert val_ratio + test_ratio < 1
            train_ratio = 1 - val_ratio - test_ratio
            print('[Warning] train_ratio is None, using all training data.')
        else:
            assert train_ratio + val_ratio + test_ratio <= 1
        indices = list(range(total_size))
        if kwargs['train_size']:
            train_size = kwargs['train_size']
        else:
            train_size = int(train_ratio * total_size)
        if kwargs['test_size']:
            test_size = kwargs['test_size']
        else:
            test_size = int(test_ratio * total_size)
        if kwargs['val_size']:
            valid_size = kwargs['val_size']
        else:
            valid_size = int(val_ratio * total_size)
        print("Final train/val/test sizes are: %d / %d / %d"%(train_size,valid_size,
                                                            test_size))
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(
            indices[-(valid_size + test_size):-test_size])
        if return_test:
            test_sampler = SubsetRandomSampler(indices[-test_size:])
    else:
        cross_num = 0
        if cross_validation == 'k-fold' or cross_validation == 'k-fold-cross-validation':
            cross_num = 0
        elif cross_validation == 'bootstrapping' or cross_validation == 'bootstrap':
            cross_num = 1
        elif cross_validation == 'leave-p-out':
            cross_num = 2
        elif cross_validation == 'leave-one-out':
            cross_num = 3
        elif cross_validation == 'monte-carlo' or cross_validation == 'monte-carlo-cross-validation':
            cross_num = 4
        
        dictdict = splitValidation(len(dataset), cross_num, cross_param, counter)
        train_sampler = SubsetRandomSampler(dictdict.get("train"))
        val_sampler  = SubsetRandomSampler(dictdict.get("val"))
        test_sampler = SubsetRandomSampler(dictdict.get("test"))
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_atom_type, batch_nbr_type, batch_nbr_dist, batch_pair_type = [],[],[],[] # MW
    batch_nbr_fea_idx_all, batch_gs_fea, batch_gp_fea, batch_gd_fea = [], [], [], []
    batch_global_fea = [] # MW
    crystal_atom_idx, batch_target = [], []
    batch_target_Fxyz = []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx,
             atom_type, nbr_type, nbr_dist, pair_type, 
             global_fea, 
             nbr_fea_idx_all, gs_fea, gp_fea, gd_fea), # MW
            target, target_Fxyz, cif_id)\
            in enumerate(dataset_list):

        # Standard features
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx)


        # additional info needed for hybridizing w/classical potenteial (i.e.
        batch_atom_type.append(atom_type) #MW
        batch_nbr_type.append(nbr_type) #MW
        batch_nbr_dist.append(nbr_dist) #MW
        batch_pair_type.append(pair_type) #MW

        # bookkeeping for crystals in batch, etc
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_target_Fxyz.append(target_Fxyz)
        batch_cif_ids.append(cif_id)

        # additional global crys features for each example
        batch_global_fea.append(global_fea) # MW

        # features needed for SpookyNet type model (i.e. spherical harmonics)
        # NOTE, all objects are lists because each Tensor entry has variable dim=0
        # due to the possible different number of neighbors for each atom environment
        batch_nbr_fea_idx_all+=[idx_map+base_idx for idx_map in nbr_fea_idx_all]
        batch_gs_fea += gs_fea
        batch_gp_fea += gp_fea
        batch_gd_fea += gd_fea

        # increment base_idx by number of atoms in this crystal
        base_idx += n_i
    try:
        stacked_Fxyz = torch.stack(batch_target_Fxyz, dim=0)
    except:
        stacked_Fxyz = None
    #print(batch_global_fea)
    #print(np.where([len(v)==2 for v in batch_global_fea]))
    #print(np.array(batch_cif_ids)[np.where([len(v)==2 for v in batch_global_fea])[0]])
    #print(torch.stack(batch_gs_fea, dim=0).shape)
    #print(torch.cat(batch_nbr_fea_idx_all, dim=0).shape)
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
            torch.cat(batch_atom_type, dim=0),
            torch.cat(batch_nbr_type, dim=0),
            torch.cat(batch_nbr_dist, dim=0),
            torch.cat(batch_pair_type, dim=0),
            torch.Tensor(batch_global_fea),
            torch.cat(batch_nbr_fea_idx_all,dim=0),# NOTE, if nonvec version, must stay list 
            torch.stack(batch_gs_fea, dim=0),      # NOTE, if nonvec version, must stay list
            torch.stack(batch_gp_fea, dim=0),      # NOTE, if nonvec version, must stay list
            torch.stack(batch_gd_fea, dim=0)),\
        torch.stack(batch_target, dim=0),\
        stacked_Fxyz,\
        batch_cif_ids


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        res = np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)
        return res

class G2Descriptor(object):
    """
    Expands interatomic distance by all G2 basis functions
    """

    def __init__(self,Rc,etas_offsets_basis=[],large=False):
        """
        Rc : float
            Radius at which interactions are ignored
        eta_offsets_basis : list of (eta, offset)
            List of basis set parameters for G2 descriptor
        """

        if not etas_offsets_basis:
            etas = [0.5,1.0,1.5]
            offsets = [1.0,2.0,3.0,4.0,5.0] 
            etas_offsets_basis = list(itertools.product(etas,offsets))
            if large:
                etas_offsets_basis += list(itertools.product([100],
                                           [2.0,2.2,2.4,2.6]))
                etas_offsets_basis += list(itertools.product([1000],
                                           [1.0,1.1,1.3,1.4,1.5]))

        self.etas_offsets_basis = etas_offsets_basis
        self.etas = np.array([tup[0] for tup in etas_offsets_basis])
        self.offsets = np.array([tup[1] for tup in etas_offsets_basis])
        self.Rc = Rc

    def row_apply(self,Rij):
        """
        Rij : float
            The interatomic distance
        """
   
        # TODO
        # Add cutoff function 
        return [np.exp(-eta * ((Rij - offset) ** 2.)/ (self.Rc ** 2.)) \
                for (eta, offset) in self.etas_offsets_basis]
        

    def expand(self,distances):
        """
        Apply BP G2 descriptors (only atom centered, so basically takes a list 
        of eta values for the filter)
        """

        # TODO
        # should be good vectorization, need to double check...
        res = np.exp(-self.etas*((distances[..., np.newaxis] - self.offsets) ** 2.)/\
                     (self.Rc ** 2.))
        
        return res



class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
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
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...
    # MW added
    ├── id0.forces
    ├── id1.forces

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, Fxyz=False, all_elems=[0],
                 max_num_nbr=12, 
                 radius=4.5, 
                 dmin=0, 
                 step=0.2,
                 random_seed=123,
                 crys_spec = None,
                 atom_spec = None,
                 csv_ext = '',
                 model_type='cgcnn',
                 K = 4,
                 njmax=75,
                 init_embed_file='atom_init.json'):
        self.root_dir = root_dir
        self.Fxyz = Fxyz
        self.all_elems = all_elems
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.dmin = dmin
        self.step = step
        self.random_seed = random_seed
        self.crys_spec = crys_spec
        self.atom_spec = atom_spec
        self.csv_ext = csv_ext
        self.model_type = model_type
        if self.model_type == 'spooky':
            self.compute_sph_harm = True
        else:
            self.compute_sph_harm = False
        self.K = K
        self.njmax = njmax # max neighs for sph_harm #TODO handle non-padding scenario of njmax=0
        self.init_embed_file = init_embed_file

        self.reload_data()

    def reload_data(self):
    
        assert os.path.exists(self.root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv'+self.csv_ext)
        assert os.path.exists(id_prop_file), 'id_prop.csv%s does not exist!'%self.csv_ext

        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        random.seed(self.random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, self.init_embed_file)
        assert os.path.exists(atom_init_file), 'initial atom emebdding file (atom_init.json) does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        #self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.gdf = G2Descriptor(Rc=self.radius,large=True)

        # if forces requested
        if self.Fxyz:
            self.id_prop_data =\
              [row+[np.loadtxt(os.path.join(self.root_dir,
                               row[0]+"_forces.csv"),delimiter=',')]\
               for row in self.id_prop_data]

        # if global crystal attributes provided
        if self.crys_spec is not None:
            #self.global_fea =\
            #  [np.loadtxt(os.path.join(self.root_dir, row[0]+"."+self.crys_spec))\
            #   for row in self.id_prop_data]
            self.global_fea = []
            for row in self.id_prop_data:
                self.global_fea.append(np.loadtxt(os.path.join(self.root_dir, row[0]+"."+self.crys_spec)))

        # if atom spcific attributes for each crystal provided
        if self.atom_spec is not None:
            # if local fea is 1 dimensional, loadtxt reads the column vec
            # as a row vec, so need to reconvert back to column vec
            #self.local_fea =\
            #  [np.loadtxt(os.path.join(self.root_dir,row[0]+"."+self.atom_spec))\
            #   for row in self.id_prop_data]
            self.local_fea = []
            for row in self.id_prop_data:
                arr = np.loadtxt(os.path.join(self.root_dir,row[0]+"."+self.atom_spec))
                if len(arr.shape)==1:
                    self.local_fea.append(arr.reshape(-1,1))
                else:
                    self.local_fea.append(arr)
                

        # if a list of elements specified, assumes these are the only 
        # elements that will be encountered (e.g. for interatomic potential
        # and adds on to it the ZBL repulsive term later on
        if self.all_elems != [0]:
            pair_elems = list(itertools.combinations_with_replacement(\
                             sorted(self.all_elems),2))
            self.pair_ind = {k: v for v, k in enumerate(pair_elems)}
        else:
            self.pair_ind = {-1: -1}

    def reset_root(self, root_dir):
        self.root_dir = root_dir
        self.reload_data()

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):

        # Target quantities
        if self.Fxyz:
            cif_id, target, target_Fxyz = self.id_prop_data[idx]
        else:
            cif_id, target = self.id_prop_data[idx]
        #print("Loading %d:"%idx, cif_id)

        # Base structure information
        #crystal = Structure.from_file(os.path.join(self.root_dir,
        #                                           cif_id+'.cif'))
        # An annoying additional step required for preserving the atomic 
        # ordering between the cif file and the Structure object
        structure = ase_read(os.path.join(self.root_dir,cif_id+'.cif'))
        crystal = Structure(structure.get_cell(),
                            structure.get_chemical_symbols(),
                            structure.get_positions(),
                            coords_are_cartesian=True)

        all_atom_types = [ELEM_DICT[crystal[i].specie.symbol]\
                          for i in range(len(crystal))]

        all_nbrs = crystal.get_all_neighbors(self.radius, 
                                             include_index=True)


        # Featurization
        ##atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
        ##                      for i in range(len(crystal))])
        #atom_fea = np.vstack([self.ari.get_atom_fea(num) for num in all_atom_types])
        #atom_fea = torch.Tensor(atom_fea)
        #all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        #nbr_fea_idx, nbr_dist = [], []
        #nbr_type, pair_type = [], [] # MW
        ##for nbr in all_nbrs:
        #for i, nbr in enumerate(all_nbrs):
        #    if len(nbr) < self.max_num_nbr:
        #        warnings.warn('{} not find enough neighbors to build graph. '
        #                      'If it happens frequently, consider increase '
        #                      'radius.'.format(cif_id))
        #        nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
        #                           [0] * (self.max_num_nbr - len(nbr)))
        #        nbr_dist.append(list(map(lambda x: x[1], nbr)) +
        #                       [self.radius + 1.] * (self.max_num_nbr -
        #                                             len(nbr)))
        #        nbr_type.append(list(map(lambda x: x[0].specie.number)) +
        #                        [0] * (self.max_num_nbr - len(nbr)))
        #        pair_type.append(\
        #          list(\
        #            map(lambda x: self.pair_ind[
        #                tuple(sorted([atom_type[i],x[0].specie.number]))], nbr))+
        #            [-1] * (self.max_num_nbr - len(nbr)))
        #    else:
        #        nbr_fea_idx.append(list(map(lambda x: x[2],
        #                                    nbr[:self.max_num_nbr])))
        #        nbr_dist.append(list(map(lambda x: x[1],
        #                                nbr[:self.max_num_nbr])))
        #        nbr_type.append(list(map(lambda x: x[0].specie.number,
        #                                nbr[:self.max_num_nbr])))
        #        pair_type.append(\
        #          list(\
        #            map(lambda x: self.pair_ind[
        #                tuple(sorted([atom_type[i],x[0].specie.number]))],
        #                                nbr[:self.max_num_nbr])))

        ## TODO need to test that pair_type created as expected
        #nbr_fea_idx, nbr_dist = np.array(nbr_fea_idx), np.array(nbr_dist)

        #nbr_fea = self.gdf.expand(nbr_dist)
        #nbr_dist = torch.Tensor(nbr_dist)
        #atom_fea = torch.Tensor(atom_fea)
        #nbr_fea = torch.Tensor(nbr_fea)
        #nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        #atom_type = torch.LongTensor(atom_type) #MW
        #nbr_type = torch.LongTensor(nbr_type) #MW
        #pair_type = torch.LongTensor(pair_type) #MW
        atom_fea, nbr_fea, nbr_fea_idx, atom_type,\
        nbr_type, nbr_dist, pair_type,\
        nbr_fea_idx_all, gs_fea, gp_fea, gd_fea =\
            self.featurize_from_nbr_and_atom_list(all_atom_types, 
                                                  all_nbrs, 
                                                  crystal,
                                                  cif_id)

        if self.crys_spec is not None:
            global_fea = list(self.global_fea[idx])
        else:
            global_fea = []

        if self.atom_spec is not None:
            local_fea = self.local_fea[idx]
            #print(cif_id, atom_fea.shape, torch.Tensor(local_fea).shape)
            atom_fea = torch.hstack([atom_fea, torch.Tensor(local_fea)])


        # return format for DataLoader
        target = torch.Tensor([float(target)])
        if self.Fxyz:
            # TODO delete this option
            target_Fxyz = torch.Tensor(target_Fxyz)
            return (atom_fea, nbr_fea, nbr_fea_idx,\
                    atom_type, nbr_type, nbr_dist, pair_type,\
                    nbr_fea_idx_all, gs_fea, gp_fea, gd_fea),\
                   target, target_Fxyz, cif_id
        else:
            if self.crys_spec is not None:
                return (atom_fea, nbr_fea, nbr_fea_idx,\
                        atom_type, nbr_type, nbr_dist, pair_type,\
                        global_fea,\
                        nbr_fea_idx_all, gs_fea, gp_fea, gd_fea),\
                       target, None, cif_id
            else:
                return (atom_fea, nbr_fea, nbr_fea_idx,\
                        atom_type, nbr_type, nbr_dist, pair_type,\
                        global_fea,\
                        nbr_fea_idx_all, gs_fea, gp_fea, gd_fea),\
                       target, None, cif_id


    def featurize_from_crystal(self,crystal):
        """
        Original code not conveniently setup to quickly process a single struct
        Will eventually need to do a lot of reworking here
        shouldn't be 2 duplicate feature initializers between this and __getitem__
        
        crystal : pymatgen.core.structure.Structure object
            an individual crystal structure to featurize for fast predictions
            outside the normal train/test infrastructure
        """
        cif_id, target, target_Fxyz = 0, None, None 

        all_atom_types = [crystal[i].specie.number\
                          for i in range(len(crystal))]

        all_nbrs = crystal.get_all_neighbors_old(self.radius, 
                                                 include_index=True)

        all_coords = crystal.coords

        atom_fea, nbr_fea, nbr_fea_idx, atom_type, nbr_type, nbr_dist, pair_type =\
            self.featurize_from_nbr_and_atom_list(all_atom_types, all_nbrs, cif_id)

        if self.Fxyz:
            return (atom_fea, nbr_fea, nbr_fea_idx,\
                    atom_type, nbr_type, nbr_dist, pair_type),\
                   target, target_Fxyz, cif_id
        else:
            return (atom_fea, nbr_fea, nbr_fea_idx,\
                    atom_type, nbr_type, nbr_dist, pair_type),\
                   target, None, cif_id

    def featurize_from_nbr_and_atom_list(self, all_atom_types, 
                                               all_nbrs, 
                                               crystal,
                                               cif_id='struct'):
        """
        all_atom_types : list of ints
            list of atomic numbers for all sites
        all_nbrs : list of list of [int, float, int]
            Zj1 is atomic number of the nbr j1
            r_ij1 is distance betwen site i and nbr j1
            ind_j1 is site index of nbr j1

                   [
                     [ # site i=0
                       [ Z_j1 , r_ij1 , ind_j1 ],
                       [ Z_j2 , r_ij2 , ind_j2 ],
                       [ ...                   ]
                     ] ,
                     [ # site i=1
                       [ Z_j1 , r_ij1 , ind_j1 ],
                       [ Z_j2 , r_ij2 , ind_j2 ],
                       [ ...                   ]
                     ]
                     ...
                   ]

        TODO: cleanup the insane list comprehensions
        TODO: break-up this function into model specific requirements
              for featurization
        """
        # Featurization
        #atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
        #                      for i in range(len(crystal))])
        atom_fea = np.vstack([self.ari.get_atom_fea(num)\
                             for num in all_atom_types])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_dist, nbr_fea_idx_all = [], [], []
        nbr_type, pair_type = [], [] # MW
        nbr_fea_idx_all, gs_fea_all, gp_fea_all, gd_fea_all = [], [], [], []

        for i, nbr in enumerate(all_nbrs):
            if len(nbr) < self.max_num_nbr:
                warnings.warn('%s did not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'%cif_id)
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_dist.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))

                # Double check the nbr_fea_dix construction
                #assert list(map(lambda x: x[0].specie.number, nbr)) +\
                #                [0] * (self.max_num_nbr - len(nbr)) ==\
                #       list(map(lambda x: all_atom_types[x[2]], nbr)) +\
                #                [0] * (self.max_num_nbr - len(nbr)) ==\

                nbr_type.append(list(map(lambda x: all_atom_types[x[2]],nbr)) +
                                [0] * (self.max_num_nbr - len(nbr)))

                # double check the nbr type computation
                #assert list(map(lambda x: self.pair_ind[
                #         tuple(sorted([all_atom_types[i],i
                #                       x[0].specie.number]))], nbr)) +\
                #         [-1] * (self.max_num_nbr - len(nbr)) ==\
                #       list(map(lambda x: self.pair_ind[\
                #         tuple(sorted([all_atom_types[i],
                #                       all_atom_types[x[2]]]))], nbr)) +\
                #         [-1] * (self.max_num_nbr - len(nbr))

                if self.all_elems != [0]:
                    pair_type.append(\
                      list(map(lambda x: self.pair_ind[
                            tuple(sorted([all_atom_types[i],all_atom_types[x[2]]]))], nbr)) +
                           [-1] * (self.max_num_nbr - len(nbr)))
                else:
                    pair_type.append([-1] * self.max_num_nbr)
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_dist.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))

                # Double check the nbr_fea_idx construction
                #assert list(map(lambda x: x[0].specie.number, 
                #                nbr[:self.max_num_nbr])) ==\
                #       list(map(lambda x: all_atom_types[x[2]], 
                #                nbr[:self.max_num_nbr]))

                nbr_type.append(list(map(lambda x: all_atom_types[x[2]],
                                        nbr[:self.max_num_nbr])))

                # double check the nbr_type construction
                #assert list(map(lambda x: self.pair_ind[\
                #        tuple(sorted([all_atom_types[i],
                #                      x[0].specie.number]))],
                #              nbr[:self.max_num_nbr])) ==\
                #       list(map(lambda x: self.pair_ind[\
                #        tuple(sorted([all_atom_types[i],
                #                      all_atom_types[x[2]]]))],
                #              nbr[:self.max_num_nbr]))
                if self.all_elems != [0]:
                    pair_type.append(\
                      list(\
                        map(lambda x: self.pair_ind[
                            tuple(sorted([all_atom_types[i],
                                          all_atom_types[x[2]]]))],
                            nbr[:self.max_num_nbr])))
                else:
                    pair_type.append([-1] * self.max_num_nbr)

                if self.compute_sph_harm:
                    # duplicate nbr_fea_idx but for ALL nbrs (no max #) 
                    nbr_fea_idx_all.append(\
                        torch.LongTensor(list(map(lambda x: x[2],nbr))+\
                                         [0] * (self.njmax - len(nbr)))
                    )
                else:
                    # dummy data now for compatibility, need to just do 
                    # a big refactor to separate the featurization
                    # strategies for different models
                    nbr_fea_idx_all.append(torch.LongTensor([0]))
                
        if self.compute_sph_harm:        
            # compute gs, gp, gd for all atoms in crystal here
            gs_fea, gp_fea, gd_fea =\
                get_harmonics_fea(crystal, all_nbrs, self.K, self.radius, self.njmax)
        else:
            # will be completely unused, but for compatibility with collate_pool
            gs_fea, gp_fea, gd_fea = [torch.zeros(0)], [torch.zeros(0)], [torch.zeros(0)]

        # TODO need to test that pair_type created as expected
        nbr_fea_idx, nbr_dist = np.array(nbr_fea_idx), np.array(nbr_dist)

        nbr_fea = self.gdf.expand(nbr_dist)
        nbr_dist = torch.Tensor(nbr_dist)
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        atom_type = torch.LongTensor(all_atom_types) #MW
        nbr_type = torch.LongTensor(nbr_type) #MW
        pair_type = torch.LongTensor(pair_type) #MW

        return (atom_fea, nbr_fea, nbr_fea_idx,\
                atom_type, nbr_type, nbr_dist, pair_type,\
                nbr_fea_idx_all, gs_fea, gp_fea, gd_fea)\


@torch.jit.script
class CIFDataFeaturizer():

    def __init__(self):
        self.foo()

    def foo(self):
        return 1
