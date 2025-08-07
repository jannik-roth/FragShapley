import pandas as pd

def create_classification_datasets(chembl, accs_of_interest):
    all_smiles_of_accs = chembl.loc[chembl.accession.isin(accs_of_interest)].nonstereo_aromatic_smiles.unique()
    pool_of_negative_samples = chembl.loc[~chembl.nonstereo_aromatic_smiles.isin(all_smiles_of_accs)].drop_duplicates(subset='nonstereo_aromatic_smiles')

    cols_to_keep = ['nonstereo_aromatic_smiles', 'accession', 'standard_type', 'chembl_cid', 'chembl_tid', 'label']

    for acc in accs_of_interest:
        positive_samples = chembl.loc[chembl.accession == acc].drop_duplicates(subset='nonstereo_aromatic_smiles')
        negative_samples = pool_of_negative_samples.sample(n=positive_samples.shape[0],
                                                           replace=False,
                                                           random_state=42,
                                                           )
        positive_samples['label'] = 1
        negative_samples['label'] = 0

        df_comb = pd.concat((positive_samples, negative_samples))
        df_comb[cols_to_keep].to_csv(f'classification/{acc}.csv')

def create_regression_datasets(chembl, accs_of_interest, min_cutoff):

    sts = ['IC50', 'Kd']
    cols_to_keep = ['nonstereo_aromatic_smiles', 'accession', 'standard_type', 'pPot_mean', 'chembl_cid', 'chembl_tid']
    for acc in accs_of_interest:
        for st in sts:
            df = chembl.loc[(chembl.accession == acc) & (chembl.standard_type == st)]
            if df.shape[0] >= min_cutoff:
                df[cols_to_keep].to_csv(f'regression/{acc}_{st}.csv', index=False)

def main():
    path_to_chembl = 'final_high_conf.tsv.gz'
    topk = 5
    min_cutoff = 500
    
    chembl = pd.read_csv(path_to_chembl,
                         delimiter='\t')
    # basic filtering
    chembl = chembl.loc[chembl.organism == 'Homo sapiens']
    chembl = chembl.dropna(subset='pPot_mean')
    # get accession of interest (largest ones)
    accs_of_interest = chembl.accession.value_counts()[:topk].index.to_list()

    # create the classification datasets
    create_classification_datasets(chembl=chembl,
                                   accs_of_interest=accs_of_interest,
                                   )
    
    # create regression datasets
    create_regression_datasets(chembl=chembl,
                               accs_of_interest=accs_of_interest,
                               min_cutoff=min_cutoff,
                               )
    
if __name__ == '__main__':
    main()