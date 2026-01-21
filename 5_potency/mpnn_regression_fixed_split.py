import pandas as pd
import lightning as L
import torch
from torch_geometric.loader import DataLoader
import argparse
import sys
import os

sys.path.append('/home/jannikroth/work/20_FragShapley/FragShapley/')
import FragShapley

import yaml

def main(param_file):

    # load params file
    with open(param_file) as params:
        args = yaml.load(params, Loader=yaml.FullLoader)
    
    # extract important parameters
    ## general
    regression_datasets = args['general']['regression_datasets']
    path_to_regression_datasets = args['general']['path_to_regression_datasets']
    smiles_column = args['general']['smiles_column']
    y_column = args['general']['y_column']
    set_column = args['general']['set_column']
    # n_cv = args['general']['n_cv']
    # random_state = args['general']['random_state']
    results_folder = args['general']['results_folder']
    ## hyperparameter optimization
    mpnnr_model_params = args['MPNN']['mpnnr_model_parameter']
    accelerator = args['MPNN']['accelerator']
    max_epochs = args['MPNN']['max_epochs']
    batch_size = args['MPNN']['batch_size']
    ## for explainability
    expected_value = args['general']['expected_value']

    # folder management
    ## check if results folder is already present to avoid over-writing results
    if os.path.isdir(results_folder):
        raise ValueError('Results folder does already exist!')
    os.mkdir(results_folder)
    ## create subfolders for crossvalidation results and models
    cv_results_folder = os.path.join(results_folder, 'cv_results')
    os.mkdir(cv_results_folder)
    model_results_folder = os.path.join(results_folder, 'models')
    os.mkdir(model_results_folder)

    # acutal calculations now

    ## initialize to save results
    rows_performance = []
    df_expl = pd.DataFrame()

    ## loop over all datasets
    for regression_dataset in regression_datasets:
        print(f'On regression dataset: {regression_dataset}')

        # load dataset
        df = pd.read_csv(os.path.join(path_to_regression_datasets, f'{regression_dataset}.csv'))

        smiles_train = df.loc[df[set_column] == 'Train'][smiles_column].to_list()
        smiles_test = df.loc[df[set_column] == 'Test'][smiles_column].to_list()
        y_train, y_test = df.loc[df[set_column] == 'Train'][y_column].to_list(), df.loc[df[set_column] == 'Test'][y_column].to_list()

    
        featurizer = FragShapley.Featurizer(input_format='smiles',
                                            output_format='graph')
        
        ds_train = FragShapley.MoleculeDataset(list_of_inputs=smiles_train,
                                                featurizer=featurizer,
                                                y=y_train)
        ds_test = FragShapley.MoleculeDataset(list_of_inputs=smiles_test,
                                                featurizer=featurizer,
                                                y=y_test)
        dl_train = DataLoader(dataset=ds_train,
                                batch_size=batch_size,)
        dl_test = DataLoader(dataset=ds_test,
                                batch_size=batch_size,)
        
        trainer = L.Trainer(accelerator=accelerator,
                            max_epochs=max_epochs,)
        model = FragShapley.MPNNRegressor(**mpnnr_model_params)
        trainer.fit(model=model,
                    train_dataloaders=dl_train,)
        # save the model
        torch.save(model,os.path.join(model_results_folder, f'model_mpnnr_{regression_dataset}.pkl'))
        
        y_pred = torch.vstack(trainer.predict(model, dl_test)).detach().cpu().numpy().squeeze()
        # performance metrics
        rows_performance.append({'model': 'MPNN',
                                 'dataset': regression_dataset,
                                 'model_params': mpnnr_model_params | {'max_epochs': max_epochs, 'batch_size': batch_size},
                                 'y_test': y_test,
                                 'y_pred': y_pred,
                                 })

        # now the explanations

        # run FragShapley
        frag_explainer = FragShapley.FragmentExplainer(model=model,
                                                        expected_value=expected_value,
                                                        fragmentation_method='BRICS',
                                                        representation='graph',
                                                        trainer=trainer,
                                                        featurizer=featurizer,
                                                        batch_size=batch_size,)
        ev_frag = frag_explainer.expected_value
        results_dicts, frag_to_atom_ids = frag_explainer.explain(smiles_test, return_atom_id_to_bits=False)

        results = {'model': 'MPNN',
                   'dataset': regression_dataset,
                   'smiles': smiles_test,
                   'y_true': y_test,
                   'y_pred': y_pred,
                   'fragExplainer_result': results_dicts,
                   'fragExplainer_expected_value': [ev_frag for _ in smiles_test],
                   'frag_to_atom_ids': frag_to_atom_ids,
                   }
        df_expl_inner = pd.DataFrame(results)
        df_expl = pd.concat((df_expl, df_expl_inner))

    df_performance = pd.DataFrame(rows_performance)
    df_performance.to_pickle(os.path.join(results_folder, 'df_performance.pkl'))
    df_expl.to_pickle(os.path.join(results_folder, 'df_explanation.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file', help='parameter file', type=str, required=True)
    args = parser.parse_args()

    print('The following parameter file will be used: ', args.param_file)
    main(param_file=args.param_file)
