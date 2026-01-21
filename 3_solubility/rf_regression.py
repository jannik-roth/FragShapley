import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from pickle import dump
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
    n_cv = args['general']['n_cv']
    n_hyperopt = args['general']['n_hyperopt']
    random_state = args['general']['random_state']
    results_folder = args['general']['results_folder']
    ## fingerprint
    fpSize = args['general']['fpSize']
    radius = args['general']['radius']
    ## hyperparameter optimization
    rfr_parameter_grid = args['random_forest']['rfr_parameter_grid']
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

        # cross validation
        cv = KFold(n_splits=n_cv,
                   shuffle=True,
                   random_state=random_state,
                   )
        # loop over all splits
        for split, (train_index, test_index) in enumerate(cv.split(df)):
            print(f'\tIn split: {split}')

            # split data and get fingerprints
            smiles_train = df[smiles_column].iloc[train_index].to_list()
            smiles_test = df[smiles_column].iloc[test_index].to_list()
            y_train, y_test = df[y_column].iloc[train_index].to_list(), df[y_column].iloc[test_index].to_list()

            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,
                                                               fpSize=fpSize,
                                                              )
            fps_train = np.stack([mfpgen.GetFingerprintAsNumPy(MolFromSmiles(sm)) for sm in smiles_train])
            fps_test = np.stack([mfpgen.GetFingerprintAsNumPy(MolFromSmiles(sm)) for sm in smiles_test])

            # set up hyper parameter optimization using GridSearchCV
            ## model
            rfr = RandomForestRegressor(random_state=random_state)
            ## inner cv for hyperparameter optimization
            inner_cv = KFold(n_splits=n_hyperopt,
                             shuffle=True,
                             random_state=random_state,
                             )
            ## gridsearch
            gridCV = GridSearchCV(estimator=rfr,
                                  param_grid=rfr_parameter_grid,
                                  scoring='neg_root_mean_squared_error', # use RMSE here
                                  refit=True, # we want to use the best estimator afterwards
                                  cv=inner_cv,
                                  )
            gridCV.fit(X=fps_train,
                       y=y_train)
            # save cv results
            pd.DataFrame(gridCV.cv_results_).to_pickle(os.path.join(cv_results_folder, f'cv_results_rfr_{regression_dataset}_split_{split}.pkl'))
            
            best_regr = gridCV.best_estimator_
            # save best model, see https://scikit-learn.org/stable/model_persistence.html#pickle-joblib-and-cloudpickle
            with open(os.path.join(model_results_folder, f'model_rfr_{regression_dataset}_split_{split}.pkl'), 'wb') as f:
                dump(best_regr, f, protocol=5)

            # performance metrics
            y_pred = best_regr.predict(fps_test)
            rows_performance.append({'model': 'RF',
                                     'dataset': regression_dataset,
                                     'split': split,
                                     'best_params': gridCV.best_params_,
                                     'train_index': train_index,
                                     'test_index': test_index,
                                     'y_test': y_test,
                                     'y_pred': y_pred,
                                     })

            # now the explanations

            # run FragShapley
            frag_explainer = FragShapley.FragmentExplainer(model=best_regr,
                                                           fingerprint_generator=mfpgen,
                                                           fragmentation_method='BRICS',
                                                           expected_value=expected_value)
            ev_frag = frag_explainer.expected_value
            # results_dicts, atom_id_to_bits, frag_to_atom_ids = frag_explainer.explain(smiles_explain)
            results_dicts, frag_to_atom_ids, atom_id_to_bits = frag_explainer.explain(smiles_test, return_atom_id_to_bits=True)

            # run shap TreeExplainer
            shap_explainer = shap.TreeExplainer(model=best_regr,
                                                data=fps_train,
                                                model_output='raw',
                                                feature_perturbation='interventional',
                                                )
            shap_expl = shap_explainer.shap_values(fps_test)
            shap_expl = [i for i in shap_expl]
            shap_ev = shap_explainer.expected_value

            results = {'model': 'RF',
                       'dataset': regression_dataset,
                       'split': split,
                       'smiles': smiles_test,
                       'y_true': y_test,
                       'y_pred': y_pred,
                       'fragExplainer_result': results_dicts,
                       'fragExplainer_expected_value': [ev_frag for _ in smiles_test],
                       'shap_result': shap_expl,
                       'shap_expected_value': [shap_ev for _ in smiles_test],
                       'atom_id_to_bits': atom_id_to_bits,
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
