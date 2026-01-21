import pandas as pd
import numpy as np

from rdkit.Chem import Draw, MolFromSmiles
from IPython.display import SVG
from collections import defaultdict

def visualize_contributions(smiles, 
                            contributions, 
                            scale=0.5,
                            legend="",
                            color_pos=(1,.5,.5),
                            color_neg=(.5,.5,1),):
    '''
        Visualizes the contribution of with circles. Uses rdKit. Draw to create a SVG. Contributions are highlighted using circles of different colours, the radius scaling accoringly to the contribution

        input:
            smiles: str, SMILES of the molecule
            contributions: np.array, array containing the contributions per atom
            scaler: float, controls the scaling of the radius of the circles drawn. Needs to be adjusted accordingly
    '''

    return visualize_contributions_from_mol(MolFromSmiles(smiles), contributions, scale, legend, color_pos, color_neg)

def visualize_contributions_from_mol(mol, 
                                     contributions, 
                                     scale=0.5,
                                     legend="",
                                     color_pos=(1,.5,.5),
                                     color_neg=(.5,.5,1),
                                     ):
    '''
        Visualizes the contribution of with circles. Uses rdKit. Draw to create a SVG. Contributions are highlighted using circles of different colours, the radius scaling accoringly to the contribution

        input:
            mol: rdkit.Mol
            contributions: np.array, array containing the contributions per atom
            scaler: float, controls the scaling of the radius of the circles drawn. Needs to be adjusted accordingly
    '''
    radii = {}
    colors = {}
    highlight_atoms = []
    for i,cont in enumerate(contributions):
        # quantize and scale the charge so that we can use it to scale the highlight radii
        #chg = (10*chg//1)/10
        if abs(cont)>1e-4:
            radii[i] = abs(cont) * scale
            highlight_atoms.append(i)
            if cont>0:
                colors[i] = color_pos
            else:
                colors[i] = color_neg

    d2d = Draw.MolDraw2DSVG(-1, -1)
    dopts = d2d.drawOptions()
    dopts.useBWAtomPalette()
    # we need to set the highlights to be circles or we'll end up with ovals
    # that fit around the atomic symbol
    dopts.atomHighlightsAreCircles = True

    # we need to provide highlightBonds=[] here to avoid having the bonds between highlighted atoms highlighted:
    d2d.DrawMolecule(mol,
                     highlightAtoms=highlight_atoms,
                     highlightAtomColors=colors,
                     highlightAtomRadii=radii,
                     highlightBonds=[],
                     legend=legend)
    d2d.FinishDrawing()
    return SVG(d2d.GetDrawingText())

def get_atom_contribution_from_result_dict(smiles, results_dict, frag_to_atom_ids):
    '''
        Converts the contribution per fragment to contirbution per atom by assining each atom in the molecule the contribution of the whole fragment
        ###ADD NORMALIZATION!!!!###

        input:
            smiles: str, SMILES of the whole molecule
            results_dict: Dict[int, float], containing the contribution of the indivdual fragments
            frag_to_atom_ids: Dict[int, List[int]], containing a mapping of each frag id to atom id

        output
            np.array, containing the contributions distributed along all atoms
    '''
    contributions = np.zeros(MolFromSmiles(smiles).GetNumAtoms())
    for frag_id in range(len(results_dict.keys())):
        for atom_id in frag_to_atom_ids[frag_id]:
            contributions[atom_id] = results_dict[frag_id]
    return contributions

# invert atom_id_to_bits to bits_to_atom_ids
def get_bits_to_atom_ids(atom_id_to_bits):
    bits_to_atom_ids = defaultdict(list)
    for k, values in atom_id_to_bits.items():
        for v in values:
            bits_to_atom_ids[v].append(k)
    return bits_to_atom_ids

def get_atom_contribution_from_shap_results(atom_id_to_bits, shap_result, bits_to_atom_ids):
    contributions = np.zeros(len(atom_id_to_bits))
    for atom_id in atom_id_to_bits.keys():
        for bit in atom_id_to_bits[atom_id]:
            contributions[atom_id] += shap_result[bit] / len(bits_to_atom_ids[bit])
    return contributions