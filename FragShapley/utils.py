from rdkit.Chem import MolFromSmiles, MolToSmiles, FragmentOnBRICSBonds, GetMolFrags
from rdkit.Chem import ReplaceSubstructs, RemoveAllHs

def remove_isotope_information_for_dummy_atoms(m):
    '''
        Removes the isotope information from dummy atoms (denoted as *). Does nothing if molecule does not have a dummy atom

        input:
            m : rdkit.Mol, Molecule

        output:
            rdkit.Mol, Molecule without isotope information at dummy atoms
    '''
    for at in m.GetAtoms():
        if at.GetSymbol() == '*':
            at.SetIsotope(0)
    return m

def remove_dummy_atoms(m):
    '''
        Removed dummy atoms (denoted as *) from molecule. Works by replacing dummy atom(s) with hydrogen(s)

        input:
            m : rdkit.Mol, Molecule
        
        output:
            rdkit.Mol, sanitized Molecule without dummy atom(s)
    '''
    dummy = MolFromSmiles('*')
    m_cleaned = ReplaceSubstructs(m, dummy, MolFromSmiles('[H]'), replaceAll=True)[0]
    return RemoveAllHs(m_cleaned, sanitize=True)

def get_BRICS_fragments_as_SMILES(smiles, remove_dummies=False):
    '''
        Fragments a SMILES using BRICS and returns the fragments as a list of SMILES. First, turns the input SMILES into rdkit.Mol.
        Followed by fragmenting using BRICS and removal of isotope information on the cleavage sites.

        input:
            smiles: str, SMILES string of molecule to fragment
            remove_dummies: bool, default False, flag to remove dummy atoms from fragments

        output:
            List[str] list of SMILES without isotope information and depending on remove_dummies with or without dummy atoms
    '''
    mol = MolFromSmiles(smiles)
    frag_ = FragmentOnBRICSBonds(mol)
    frags = GetMolFrags(frag_,
                        asMols=True)
    frags_cleaned = [remove_isotope_information_for_dummy_atoms(m) for m in frags]
    if remove_dummies:
        frags_cleaned = [remove_dummy_atoms(m) for m in frags_cleaned]
        
    return [MolToSmiles(m) for m in frags_cleaned]

def filter_n_attachment(smiles, n):
    '''
        Filter for number of attachment points in fragment  (equal to number of dummy atoms)

        input:
            smiles: SMILES string of molecule to check for number of attachment points
            n: number of attachments points
        ouput:
            bool, True if number of attachment points is equal to n, else False
    '''
    if smiles.count('*') == n:
        return True
    else:
        return False