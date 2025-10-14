# FragShapley: Exact Shapley Values for Fragments of Molecules

FragShapley implements the calculation of exact Shapley values for molecular structures. The obtained Shapley values can be used for the explanation of predictions in various tasks.

## Related Literature

1. [Chemistry-intuitive explanation of graph neural networks for molecular property prediction with substructure masking](https://doi.org/10.1038/s41467-023-38192-3)

## Datasets
Possible datasets that can be used for the study

1. ChEMBL: potency prediction (regression), activity prediction (classification, active vs. random compounds) -> Maybe look for alternative
2. ESOL, solubility (regression)
3. Ames data, mutagenicity (classification, might be able to use the data from [this study](https://doi.org/10.1021/acs.chemrestox.4c00466))

## Ideas for Analysis

1. Comparison with existing methods (shap package, EdgeShaper)
2. "Optimization" of compounds: replace R-groups which have a negative contribution with groups that have a positive one
3. Detection of toxicophores (for the Ames dataset)
4. Analysis of sampling efficiency: Do we need to sample all possible subsets or is it enough to only sample the layer 1 coalitions?

## Implementation

### Molecular Representations
The method is currently implemented for the following molecular representations:

1. Graphs
2. Morgan Fingerprints (folded)

### Supported Models
The method supports the following models currently:

1. scikit-learn regressors/classifiers
2. GCN