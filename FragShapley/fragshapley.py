
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.special import factorial
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles, FragmentOnBRICSBonds, GetMolFrags
from functools import partial

from abc import ABC
import torch
from torch_geometric.loader import DataLoader
from .data import GraphDataset


class FragmentRepresentationHelper(ABC):
    """
    Helper Class to generate the represenation of coalitions of fragments
    """
    def __init__(self,
                 smiles,
                 fragmentation_method):
        super().__init__()
        
        self._smiles = smiles
        self._mol = MolFromSmiles(smiles)
        self._fragmentation_method = fragmentation_method
        
        # fragment molecule
        if self._fragmentation_method.lower() == 'brics':
            self._fragment_molecule_BRICS()
        else:
            raise ValueError(f"I don't know the fragmentation method: {self._fragmentation_method}")

    def _fragment_molecule_BRICS(self):
        """
        Fragments the molecule according to BRICS rules and saves the atom ids of each fragment
        """
        mol_atom_ids = []
        frag_mols = FragmentOnBRICSBonds(mol=self._mol)
        self.fragments = GetMolFrags(frag_mols,
                                     asMols=True,
                                     sanitizeFrags=False,
                                     fragsMolAtomMapping=mol_atom_ids)
        self.n_frags = len(self.fragments)
        # generate dict to save the atom ids for each fragment
        self.frag_to_atom_ids = {}
        for idx, tup in enumerate(mol_atom_ids):
            self.frag_to_atom_ids[idx] = [item for item in tup if item < self._mol.GetNumAtoms()] # filter out all of the dummy atoms

    def generate_fragment_representation(self, frag_idxs):
        """
        Function to generate the represenation of coalitions of fragments. Needs to be implemented
        for every kind of representation
        """
        raise NotImplementedError()

class FingerprintFragmentRepresentationHelper(FragmentRepresentationHelper):
    """
    Class to generate the represenation of coalitions of fragments using Fingerprints (Morgan, folded)
    """
    def __init__(self,
                 smiles,
                 fragmentation_method,
                 fingerprint_generator,
                 ):
        super().__init__(smiles=smiles,
                         fragmentation_method=fragmentation_method,
                         )
        
        self._fpgen = fingerprint_generator
        self._generate_atom_bit_info()
    
    def _generate_atom_bit_info(self):
        """
        Generates the atom bit info which maps every atom id to the corresponding bits in the fingerprint
        """
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        self.full_fp = self._fpgen.GetFingerprintAsNumPy(mol=self._mol,
                                                         additionalOutput=ao,
                                                         )
        self.atom_id_to_bits = defaultdict(list)
        for k, vs in ao.GetBitInfoMap().items():
            for v in vs:
                self.atom_id_to_bits[v[0]].append(k)

    def generate_fragment_representation(self, frag_idxs):
        """
        Generates the Fingerprint representation of the coalition of fragments
        """
        # start with empty fingperint
        frag_fp = np.zeros_like(self.full_fp)
        # collect all bits to be turned on (=1)
        idxs_to_turn_on = []
        for frag_idx in frag_idxs:
            atom_ids_in_frag = self.frag_to_atom_ids[frag_idx]
            for atom_id_in_frag in atom_ids_in_frag:
                idxs_to_turn_on += self.atom_id_to_bits[atom_id_in_frag]
        frag_fp[idxs_to_turn_on] = 1.0
        return frag_fp
    
class GraphFragmentRepresentationHelper(FragmentRepresentationHelper):
    """
    Class to generate the represenation of coalitions of fragments using Graphs
    """
    def __init__(self, smiles, fragmentation_method, featurizer):
        """
        initilization requires a featurizer, which needs to be the same as the one which was used
        to train the initial model
        """
        super().__init__(smiles, 
                         fragmentation_method)
        self.featurizer = featurizer
        self.full_graph = featurizer.featurize_single(input=smiles,
                                                      y=None)

    def generate_fragment_representation(self, frag_idxs):
        """
        Generates the graph representation of the coalition of fragments
        """
        # simply mask all the nodes which are NOT in the coalition of fragments
        mask = np.zeros(self.full_graph.num_nodes, dtype=bool)
        for frag_idx in frag_idxs:
            mask[self.frag_to_atom_ids[frag_idx]] = True  # only keep the fragments
        sg = self.full_graph.subgraph(torch.tensor(mask))
        return sg

def proba_to_logit(proba, eps=1e-6):
    """
    Converts probabilities to logits space. Uses eps to avoid division by zero
    """
    return np.log(proba/(1. - proba + eps) + eps) # to avoid division by zero, and to avoid log(0)

class FragmentExplainer():
    """
    Class to calculate the Shapley values of fragments

    model: model to be used for the explanation
    expected_value: float or "empty", if float: uses that number as value for empty coalition,
                    if "empty": uses the output of empty graph or empty fingerprint as value for empty coalition
    fragmentation_method: method used for fragmentation
    representation: type of representation, either "fp" or "graph"
    fingerprint_generator: fingerprint_generator to be used of representation is "graph"
    trainer: lightning Trainger of LightningModule is used as model
    featurizer: featurizer if LightningModule is used
    batch_size: batch size to be used during the inference
    """
    def __init__(self,
                 model,
                 expected_value,
                 fragmentation_method = 'BRICS',
                 representation = 'fp',
                 fingerprint_generator = None,
                 trainer=None,
                 featurizer=None,
                 batch_size=8,
                 ):
        
        # check which type of model here
        if 'sklearn' in str(type(model)):
            self.model = model
            self.model_origin = 'sklearn'
            self.model_predict = self.predict_sklearn_helper # helper function
            # get model_type
            if 'Regressor' in self.model.__repr__():
                self.task = 'regression'
            elif 'Classifier' in self.model.__repr__():
                self.task = 'classification'
                self.model_predict_proba = self.predict_proba_sklearn_helper # use the probability to predict 1
                assert self.model.n_classes_ == 2, "Only implemented for binary classification tasks!"
            else:
                raise ValueError("I can not identify the task (Regression/Classification)!")
        # add Graph models here please
        elif 'GCN' in str(type(model)):
            self.model = model
            self.model_origin = 'torch'
            self.model_predict = self.predict_pytorch_geometric_helper # helper function
            if 'Regressor' in self.model.__repr__():
                self.task = 'regression'
            elif 'Classifier' in self.model.__repr__(): # classifier outputs proba (through predict step with sigmoid as final step)
                self.task = 'classification'
                self.model_predict_proba = self.predict_pytorch_geometric_helper
            else:
                raise ValueError("I can not identify the task (Regression/Classification)!")
        elif 'MPNN' in str(type(model)):
            self.model = model
            self.model_origin = 'torch'
            self.model_predict = self.predict_pytorch_geometric_helper # helper function
            if 'Regressor' in self.model.__repr__():
                self.task = 'regression'
            elif 'Classifier' in self.model.__repr__(): # classifier outputs proba (through predict step with sigmoid as final step)
                self.task = 'classification'
                self.model_predict_proba = self.predict_pytorch_geometric_helper
            else:
                raise ValueError("I can not identify the task (Regression/Classification)!")
        else:
            raise ValueError('I can not work with this model :(')
        

        self.fragmentation_method = fragmentation_method
        self.representation = representation
        self.batch_size = batch_size

        if self.representation == 'fp':
            # check if everything is available
            if fingerprint_generator is None:
                raise ValueError('Requires Fingerprint Generator!')
            self.fingerprint_generator = fingerprint_generator
            self.fp_size = self.fingerprint_generator.GetOptions().fpSize
            # prepare the helper function to be used later
            self.helper_func = partial(FingerprintFragmentRepresentationHelper,
                                       fragmentation_method=self.fragmentation_method,
                                       fingerprint_generator=self.fingerprint_generator,
            )
        elif self.representation == 'graph':
            # check if everything is available
            if trainer is None:
                raise ValueError('Requires Trainer!')
            self.trainer = trainer
            if featurizer is None:
                raise ValueError('Requires Featurizer')
            self.featurizer = featurizer
            # prepare the helper function to be used later
            self.helper_func = partial(GraphFragmentRepresentationHelper,
                                       fragmentation_method=self.fragmentation_method,
                                       featurizer=featurizer)


        # now deal with the expected value
        self.expected_value = None
        if type(expected_value) == float:
            # if it's a float, simply use that number
            self.expected_value = expected_value
        elif type(expected_value) == str:
            if expected_value.lower() == 'empty':
                # now we compute the expected value as the output of our model using an empty input
                if self.representation == 'graph':
                    # for graphs, we use an empty graph
                    eg = self.model.get_empty_graph()
                    out = self.model_predict(inputs=[eg], coalitions=[[0, 1]]) # set coalitions to be of length != 0 to make sure it does not get masked
                    self.expected_value = out[0]
                elif self.representation == 'fp':
                    # construct empty fingerprint
                    empty_fp = np.zeros(shape=(1, self.fp_size))
                    if self.task == 'regression':
                        out = self.model_predict(empty_fp, coalitions=[[0, 1]])
                        self.expected_value = out[0]
                    elif self.task == 'classification':
                        out = self.model_predict_proba(empty_fp, coalitions=[[0, 1]])
                        self.expected_value = out[0]
            else:
                raise ValueError(f"Unrecognized keyword: {expected_value}")
        else:
            raise ValueError(f"Unrecognized input for expected value: {expected_value}")
        
        # convert to logit space for classification tasks
        if self.task == 'classification':
            self.expected_value_logit = proba_to_logit(self.expected_value)

    def explain_single_row(self, smiles, return_atom_id_to_bits=False):
        """
        Explains a single SMILES string
        """

        # initialize the helper function which takes care of fragmentation and generation of
        # fragment representation
        helper = self.helper_func(smiles=smiles)

        # need to loop over all possible subsets now
        all_feats_ids = list(range(helper.n_frags))

        # gather the representations as well as the coalitions
        repr_w, repr_wo = [], [] # representations with and without the feature of interest
        coal_w, coal_wo = [], [] # coalitions with and without the feature of interest
        ks = [] # kernel values
        feats = [] # feature of interest

        # loop over all features
        for feat_id in all_feats_ids:
            feats_wo = all_feats_ids.copy()
            feats_wo.remove(feat_id)
            # now over all subset sizes
            for n in range(len(feats_wo)+1):
                # now over all combinations of the subsets of size n
                for S in combinations(feats_wo, n):
                    feats.append(feat_id)
                    S_list = list(S)
                    ks.append(self.kernel(s=len(S_list), n=len(all_feats_ids)))
                    repr_wo.append(helper.generate_fragment_representation(frag_idxs=S_list))
                    coal_wo.append(S_list)
                    repr_w.append(helper.generate_fragment_representation(frag_idxs=S_list + [feat_id]))
                    coal_w.append(S_list + [feat_id]) 
        
        # predict together to make it more efficient instead of doing it inside the loop above
        if self.task == 'regression':
            preds_wo = self.model_predict(repr_wo, coal_wo)
            preds_w = self.model_predict(repr_w, coal_w)
        elif self.task == 'classification':
            probas_wo = self.model_predict_proba(repr_wo, coal_wo)
            probas_w = self.model_predict_proba(repr_w, coal_w)
            # convert to logits
            preds_wo = proba_to_logit(probas_wo)
            preds_w = proba_to_logit(probas_w)
        
        # now assemble results       
        results_dict = defaultdict(float)
        for feat, k, pred_w, pred_wo in zip(feats, ks, preds_w, preds_wo):
            results_dict[feat] += (k * (pred_w - pred_wo))

        # return also atom_to_id_bits if using a fingerprint representation
        if return_atom_id_to_bits:
            return results_dict, helper.frag_to_atom_ids, helper.atom_id_to_bits
        
        return results_dict, helper.frag_to_atom_ids
        
    def explain(self, list_of_smiles, return_atom_id_to_bits=False):
        """
        Function to explain a list of smiles

        Essentially loops over explain_single_row for each input, accumulates the individual outputs
        """
        res_dicts, frag_maps, atom_id_to_bits = [], [], []
        for smiles in list_of_smiles:
            out = self.explain_single_row(smiles=smiles, return_atom_id_to_bits=return_atom_id_to_bits)
            res_dicts.append(out[0])
            frag_maps.append(out[1])
            if return_atom_id_to_bits:
                atom_id_to_bits.append(out[2])

        if return_atom_id_to_bits:
            return res_dicts, frag_maps, atom_id_to_bits
        return res_dicts, frag_maps
        
    def kernel(self, s, n):
        """
        Shapley value Kernel
        """
        numerator = factorial(n - 1 - s) * factorial(s)
        denominator =  factorial(n)
        return numerator / denominator
    
    def predict_pytorch_geometric_helper(self, inputs, coalitions):
        """
        Helper function for torch models

        Gets the output of the inputs, and replaces the output with the expected value if the size of the coalition is zero
        """
        ds = GraphDataset(list_of_graphs=inputs)
        dl = DataLoader(dataset=ds,
                        batch_size=self.batch_size,
                        )
        out = self.trainer.predict(self.model, dl)
        out = torch.vstack(out).detach().cpu().numpy()[:, 0] # need to get rid of additional dimension
        return FragmentExplainer.mask_empty_coalition_value(out, coalitions, self.expected_value)
    
    def predict_sklearn_helper(self, inputs, coalitions):
        """
        Helper function for sklearn models, simplifies the predict function (regression)

        Gets the output of the inputs, and replaces the output with the expected value if the size of the coalition is zero
        """
        out = self.model.predict(inputs)
        print(inputs, out)
        return FragmentExplainer.mask_empty_coalition_value(out, coalitions, self.expected_value)
    
    def predict_proba_sklearn_helper(self, inputs, coalitions):
        """
        Helper function for sklearn models, simplifies the predict_proba function (classification)

        Gets the output of the inputs, and replaces the output with the expected value if the size of the coalition is zero
        """
        out = self.model.predict_proba(inputs)[:, 1] # only take proba wrt to positive prediction here
        return FragmentExplainer.mask_empty_coalition_value(out, coalitions, self.expected_value)

    @staticmethod
    def mask_empty_coalition_value(out, coalitions, expected_value):
        """
        Replaces the output value with the expected value if the size of the coalition is zero.
        This is due to the fact that the model returns nothing if only an empty graph is provided and the batch size is zero.
        Then we need to return the expected value. If the batch size is larger than zero, an empty graph returns a value which
        is replaced with the expected value here.
        """
        size_coalitions = np.array([len(c) for c in coalitions])
        if len(out) == 0:
            return np.array([expected_value])
        else:
            out[size_coalitions == 0] = expected_value
            return out