# Explanations for Molecular Property Predictions Using Shapley Values of Fragments

This repository contains the code for the publication "Explanations for Molecular Property Predictions Using Shapley Values of Fragments".

## Structure of the Repository

- `0_datasets/` contains the files for the preparation of the data sets as well as all as the final regression and classification data sets
- `3_solubility/` contains the scripts for running the models on the solubility dataset with the respecitve results
- `4_mutagenicity/` contains the scripts for running the models on the mutagenicity dataset with the respecitve results
- `5_potency/` contains the scripts for running the models on the potency dataset with the respecitve results
- `6_analysis/` contains the analysis scripts for the all data sets and models
  - `figures/` contains the figures created by running the analysis scripts
    - `final_figures/` contains the final figures used in the publication (assembled externally with a vector graphics editor)
- `requirements.txt` is the file created by running `pip freeze > requirements.txt` with the venv activated. Can be used to create a new virtual environment.
- `FragShapley/` contains all the code to run the fragment-based Shapley value approach introduced in the publication.
  - `data.py` contains functions for data handling
  - `fragshapley.py` implements the main Shapley value calculation
  - `models.py` contains the implemented models and helper functions
  - `utils.py` utility functions for analysis
  - `visualization.py` scripts for visualization of molecules and fragments
## Usage

The usage of the provided code for the generation of fragemnt-level Shapley values is demonstrated in `3_solubility/gcn_regression.py` (regression, GCN), `3_solubility/rf_regression.py` (regression, RF) and `4_mutagenicity/gcn_classification.py` (classification, GCN) and `4_mutagenicity/rf_classification.py` (classification, RF). 

Here is a code snippet for a scikit-learn model:
```python
  import FragShapley
  # requires
  # model (here scikit-learn model), already trained
  # mfpgen FingerprintGenerator64 object from rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator
  
  expected_value = 'empty' # can also be float

  frag_explainer = FragShapley.FragmentExplainer(model=model,
                                                 fingerprint_generator=mfpgen,
                                                 fragmentation_method='BRICS',
                                                 expected_value=expected_value,)
  ev_frag = frag_explainer.expected_value # get expected value

  # obtain explanations
  results_dicts, frag_to_atom_ids, atom_id_to_bits = frag_explainer.explain(smiles_test, return_atom_id_to_bits=True)
```

Here is an example for a GCN:
```python
  import Lightning as L
  import FragShapley
  # requires
  # model, trained GCN model with the function get_empy_graph to obtain an empty graph
  trainer = L.Trainer(accelerator='auto')
  expected_value = 'empty' # can also be float
  batch_size = 8
  featurizer = FragShapley.Featurizer(input_format='smiles', output_format='graph')

  frag_explainer = FragShapley.FragmentExplainer(model=model,
                                                 expected_value=expected_value,
                                                 fragmentation_method='BRICS',
                                                 representation='graph',
                                                 trainer=trainer,
                                                 featurizer=featurizer,
                                                 batch_size=batch_size,)
  ev_frag = frag_explainer.expected_value # get expected value
  
  # obtain explanations
  results_dicts, frag_to_atom_ids = frag_explainer.explain(smiles_test, return_atom_id_to_bits=False)
```

For full examples, check out the respective scripts!