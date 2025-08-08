# FragShapley: Exact Shapley Values for Fragments of Molecules

FragShapley implements the calculation of exact Shapley values for molecular structures. The obtained Shapley values can be used for the explanation of predictions in various tasks.

## Related Literature

1. [Chemistry-intuitive explanation of graph neural networks for molecular property prediction with substructure masking](https://doi.org/10.1038/s41467-023-38192-3)

## Datasets
Possible datasets that can be used for the study

1. ChEMBL: potency prediction (regression), activity prediction (classification, active vs. random compounds)
2. ESOL, solubility (regression)
3. Ames data, mutagenicity (classification, might be able to use the data from [this study](https://doi.org/10.1021/acs.chemrestox.4c00466))

## Implementation

### Molecular Representations
The method is currently implemented for the following molecular representations:

1. Graphs
2. Morgan Fingerprints (folded)

### Supported Models
The method supports the following models currently:

1. scikit-learn regressors/classifiers
2. GCN

### Models to implements
The following model types are planned to be implemented

1. MPNN
2. different variants of GNNs