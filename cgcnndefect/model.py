from __future__ import print_function, division
import random
import torch
import torch.nn as nn
import itertools
import numpy as np
from typing import Tuple,List

from .potentials import energyZBL
#from .data import CIFDataFeaturizer

@torch.jit.script
class CIFDataFeaturizer(object):
    def __init__(self, name:str):
        self.name = name
    def foo(self):
        print("dict:"+self.name)
        ind = [{2:[0,0]}, {1:[1,1]}, {0:[2,2]}]
        # map doesn't seem to be supported
        #print(list(map(lambda x: x[1],ind)))
        

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, Fxyz=False, all_elems=[0],
                 global_fea_len=0, new_output_dim=1):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        classification: bool
          If true, perform classification instead of regression
        Fxyz : bool
          Include forces as an additional training target
        new_output_dim: int
          Dimension of the new output layer
        """
        super(CrystalGraphConvNet, self).__init__()
        self.dataset1 = CIFDataFeaturizer("name")

        self.classification = classification
        self.Fxyz = Fxyz
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len + global_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, new_output_dim)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.logsoftmax = None
            self.dropout = None

        if self.Fxyz:
            if n_h > 1:
                self.Fxyz_fcs = nn.ModuleList([nn.Linear(atom_fea_len, 
                                                         atom_fea_len)
                                               for _ in range(n_h - 1)])
                self.Fxyz_softpluses = nn.ModuleList([nn.Softplus()
                                                     for _ in range(n_h - 1)])
            self.conv_to_fc_F = nn.Linear(atom_fea_len, atom_fea_len)
            self.fc_F_out = nn.Linear(atom_fea_len, 3)
        else:
            self.Fxyz_fcs = None
            self.Fxyz_softpluses = None
            self.conv_to_fc_F = None
            self.fc_F_out = None

    def forward(self, atom_fea: torch.Tensor, 
                      nbr_fea: torch.Tensor, 
                      nbr_fea_idx: torch.Tensor, 
                      crystal_atom_idx: List[torch.Tensor],
                      atom_type: torch.Tensor, 
                      nbr_type: torch.Tensor, 
                      nbr_dist: torch.Tensor, 
                      pair_type: torch.Tensor,
                      global_fea: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass

        Parameters
        ----------
        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx in the batch
        Returns
        -------
        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution
        """
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        if self.Fxyz:
            pass
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(
            self.conv_to_fc_softplus(torch.cat([crys_fea, global_fea], dim=1))
        )
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            pass
        if self.Fxyz:
            return [torch.tensor([0]), torch.tensor([0])]
        else:
            return [out]

    @torch.jit.export
    def compute_repulsive_ener(self, crystal_atom_idx, atom_type, 
                                     nbr_type, nbr_dist):  
        """
        ZBL energy: A physics-based repulsive potential
        that does NOT require any fitted parameters

        Parameters
        ----------

        crystal_atom_idx: list of torch.LongTensor of length N0
          len([torch.LongTensor shape (N) , ... ]) == N0
          Mapping from the crystal idx to atom idx in the batch
          e.g. [ LongTensor([0,1]), LongTensor([2,3]), .... ]
        atom_type : torch.LongTensor shape(N,1)
            Atomic number of each element/node
        nbr_type : torch.LongTensor shape (N,M)
            Atomic number of each neighbor
        nbr_dist : torch.Tensor shape (N,M)
            Neighbor distance

        Returns
        ----------

        crys_rep_ener : torch.Tensor shape (N,)
            The summed ZBL repulsive energy term (pooled) for each crystal 
        """
             


        # Note if atom_type and nbr_types are populated with 0's
        # the ZBL energy will evaluate to zero
        Zi = torch.unsqueeze(atom_type,dim=1).expand(nbr_type.shape)
        assert Zi.shape == nbr_type.shape == nbr_dist.shape
        eZBL = energyZBL(Zi,nbr_type,nbr_dist)
        crys_rep_ener, crys_size = self.direct_ener_pooling(eZBL,crystal_atom_idx)

        # must divide by two for double counting of all pairs, and normalize
        # to the per atom total energy 
        crys_rep_ener = (crys_rep_ener/2)/\
                         torch.unsqueeze(torch.tensor(crys_size),dim=1)

        return crys_rep_ener

    @torch.jit.export
    def direct_ener_pooling(self, pw_ener : torch.Tensor, 
                                  crystal_atom_idx : List[torch.Tensor]):
        # yields repulsive energy of that atomic environment
        atom_ener = torch.sum(pw_ener,dim=1,keepdim=True)
        # sums the repulsive energy for each atomic env in a crystal,
        # across all crystals
        crystal_ener = [torch.sum(atom_ener[idx_map],dim=0,keepdim=True)
                        for idx_map in crystal_atom_idx]
        # crystal_sizes = [len(idx_map) for idx_map in crystal_atom_idx]
        crystal_sizes = [idx_map.shape[0] for idx_map in crystal_atom_idx]
        return torch.cat(crystal_ener, dim=0), crystal_sizes


    @torch.jit.export
    def pooling(self, atom_fea : torch.Tensor, 
                      crystal_atom_idx : List[torch.Tensor]):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
          Must be a list of tensors since each idx_map is 
            tensor of different size (number of atoms in that crystal)
        """
#        assert torch.sum(torch.tensor([len(idx_map) for idx_map in\
 #           crystal_atom_idx])) == atom_fea.data.shape[0]

        #normal pooling
  #      summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
 #                     for idx_map in crystal_atom_idx]

        # for defect, we are really only interested with the feature
        # vector of the node that would become the defect
        #print([idx_map[0] for idx_map in crystal_atom_idx])
        summed_fea = [torch.index_select(atom_fea,0,idx_map[0])\
                      for idx_map in crystal_atom_idx]
        #print(summed_fea)
        #summed_fea = [atom_fea[idx_map[0]] for idx_map in crystal_atom_idx]

        #return torch.cat(summed_fea, dim=0)
        # DropOut Pooling Function 

 #       summed_fea = [atom_fea[idx_map[random.randint(0, len(idx_map)-1)]]  

#              for idx_map in crystal_atom_idx] 

 #       return torch.stack(summed_fea, dim=0) 

        # Max Pooling 

#        summed_fea = [torch.max(atom_fea[idx_map], dim=0, keepdim=True)[0]  

 #             for idx_map in crystal_atom_idx] 

        return torch.cat(summed_fea, dim=0)
