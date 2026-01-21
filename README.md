# Explanations for Molecular Property Predictions Using Shapley Values of Fragments

This repository contains the code for the ??? "Explanations for Molecular Property Predictions Using Shapley Values of Fragments".

## Structure of the Repository

- `0_datasets/` contains the files for the preparation of the data sets as well as all as the final regression and classification data sets
- `3_solubility/` contains the scripts for running the models on the solubility dataset with the respecitve results
- `4_mutagenicity/` contains the scripts for running the models on the mutagenicity dataset with the respecitve results
- `5_potency/` contains the scripts for running the models on the potency dataset with the respecitve results
- `6_analysis/` contains the analysis scripts for the all data sets and models
  - `figures/` contains the figures created by running the analysis scripts
    - `final_figures/` contains the final figures used in the publication (assembled externally with a vector graphics editor)
- `requirements.txt` is the file created by running `pip freeze > requirements.txt` with the venv activated. Can be used to create a new virtual environment.