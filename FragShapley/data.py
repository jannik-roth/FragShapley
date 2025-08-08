from abc import ABC
import pandas as pd
from typing import Any
from functools import partial
import torch
from torch.utils.data import Dataset
from torch_geometric.utils.smiles import from_smiles
from rdkit.Chem import MolFromSmiles, MolToSmiles

ALLOWED_INPUT_FORMAT = ['smiles'] #, 'mol']
ALLOWED_OUTPUT_FORMAT = ['graph'] #, 'folded ecfp', 'maccs']

def SMILESToMol(smiles, **kwargs):
    """
    Converts a SMILES to rdkit.Mol
    """
    m = MolFromSmiles(smiles, **kwargs)
    if m is None:
        raise UserWarning(f'Could not convert SMILES {smiles} to a valid rdkit molecule!')
    return m

def MolToMol(mol, **kwargs):
    """
    Converts rdkit.Mol to rdkit.Mol

    Only for consistency in the implementation
    """
    return mol

# def MolToFoldedECFP(mol, mfpgen, y=None):
#     return torch.from_numpy(mfpgen.GetFingerprintAsNumPy(mol)).float()

# def MolToMACCSKey(mol, y=None):
#     return torch.from_numpy(np.array(MACCSkeys.GenMACCSKeys(mol))).float()

def MolToGraph(mol,
               with_hydrogen: bool = False,
               kekulize: bool = False,
               y=None,
               ):
    """
    Converts rdkit.Mol to a torch_geometric compatible graph
    """
    # need to convert to SMILES first again :/
    smiles = MolToSmiles(mol)
    # use torch_geometric function for simplicity
    data = from_smiles(smiles,
                       with_hydrogen=with_hydrogen,
                       kekulize=kekulize)
    data.y = y # add the label
    return data
            
class Featurizer(ABC):
    """
    Converts a Molecule from Input Format to Output Format
    """
    def __init__(self, input_format: str='smiles', input_kwargs={}, output_format: str='graph', output_kwargs={}, ignore_rdkit_warnings=True):
        super().__init__()

        if ignore_rdkit_warnings:
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

        # check input format
        self.input_format = input_format.lower()
        if not self.input_format in ALLOWED_INPUT_FORMAT:
            raise UserWarning(f'I do not know the input format: {self.input_format}!')
        # check output format
        self.output_format = output_format.lower()
        if not self.output_format in ALLOWED_OUTPUT_FORMAT:
            raise UserWarning(f'I do not know the input format: {self.output_format}!')
        
        # define functions to convert from input to rdkit.Mol
        if self.input_format.lower() == 'smiles':
            self.to_mol = partial(SMILESToMol, **input_kwargs)
        elif self.input_format.lower() == 'mol':
            self.to_mol = partial(MolToMol, **input_kwargs)
        else:
            raise UserWarning(f"I don't know the input_format: {self.input_format}")
        
        # define functions to convert from rdkit.Mol to output
        if self.output_format.lower() == 'graph':
            self.to_output = partial(MolToGraph, **output_kwargs)
        # elif self.output_format.lower() == 'folded ecfp':
        #     self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(**output_kwargs)
        #     self.to_output = partial(MolToFoldedECFP, mfpgen=self.mfpgen)
        # elif self.output_format.lower() == 'maccs':
        #     self.to_output = MolToMACCSKey
        else:
            raise UserWarning(f"I don't know the output_format: {self.output_format}")
    
    def to_mols(self, inputs):
        """
        Converts inputs to rdkit.Mol using the appropriate self.to_mol function
        """
        return [self.to_mol(input) for input in inputs]

    def to_outputs(self, mols, ys):
        """
        Converts rdkit.Mol to output format
        """
        return [self.to_output(mol=mol, y=y) for mol, y in zip(mols, ys)]
    
    # these are not really necessary but convenient to have...
    def featurize_single(self, input, y=None):
        """
        Converts a single input to output
        """
        mol = self.to_mol(input)
        return self.to_output(mol=mol, y=y)
    
    def featurize(self, inputs, ys):
        """
        Converts list of inputs and targets to desired outputs
        """
        return [self.featurize_single(input, y) for input, y in zip(inputs, ys)]

class MoleculeDataset(Dataset):
    """
    Dataset for Molecules

    Uses a list of different input types, a Featurizer, and the target values to create
    a torch compatible dataset which can be loaded into a dataloader. The featurizer con-
    verts the inputs to the appropriate output format
    """
    def __init__(self,
                 list_of_inputs: list,
                 featurizer=Featurizer(),
                 y=None):
        
        super().__init__()

        # add zeros if no target values are given
        if y is None:
            y = [0.] * len(list_of_inputs)
        self.y = torch.tensor(y)

        self.list_of_inputs = list_of_inputs

        # converts the input so moleculues and the approprate representation
        self.mols = featurizer.to_mols(self.list_of_inputs) # save mols too
        self.x = featurizer.to_outputs(self.mols, self.y)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, input_col: str, featurizer= Featurizer(), y_col: str = None):
        """
        Initilizes MoleculeDataset from a pandas DataFrame
        """
        list_of_inputs = df[input_col].to_list()
        if y_col is None:
            y = None
        else:
            y = df[y_col].to_list()
        return cls(list_of_inputs=list_of_inputs, featurizer=featurizer, y=y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index) -> Any:
        return self.x[index], self.y[index]

class GraphDataset(Dataset):
    def __init__(self,
                 list_of_graphs: list,
                 y=None):
        """
        A simple torch_geometric compatible Dataset constructed from a list of graphs and target values
        
        """
        super().__init__()

        if y is None:
            y = [0.] * len(list_of_graphs)
        self.y = torch.tensor(y)

        self.list_of_graphs = list_of_graphs

    def __len__(self):
        return len(self.list_of_graphs)
    
    def __getitem__(self, index):
        return self.list_of_graphs[index], self.y[index]