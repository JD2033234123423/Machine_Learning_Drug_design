#!/usr/bin/env python3

# imports
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
import torch.nn.functional as F


# sample smiles
smiles_str = 'CC(=O)Oc1ccccc1C(=O)O'

# Convert the SMILES string to a molecular graph representation
mol = Chem.MolFromSmiles(smiles_str)
mol = Chem.AddHs(mol)  # add hydrogens to the molecule
AllChem.EmbedMolecule(mol)  # generate 3D coordinates for the molecule
graph = rdkit.Chem.rdmolops.GetAdjacencyMatrix(mol)  # get the adjacency matrix

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, A, X):
        H = F.relu(self.W1(torch.matmul(A, X)))
        Y = self.W2(torch.matmul(A, H))
        return Y