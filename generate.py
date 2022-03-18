import json
import os
import pickle
import sys
from time import time
from time import gmtime, strftime

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scoring_functions import ADScoringFunction, score
from rdkit import Chem 

def generate(dataset,
             n_estimators,
             seed,
             optimizer,
             base_results, qsar_features, ad, ad_features, multiple_ads):
    """
    Args:
        - dataset: which dataset to use
        - n_estimators: number of trees in the Random Forest
        - seed: which random seed to use
        - opt_name: which optimizer to use (graph_ga or lstm_hc)
        - optimizer: guacamol molecular optimizer
        - base_results: Directory where results are stored.
    """

    # Results might not be fully reproducible when using pytorch
    # https://pytorch.org/docs/stable/notes/randomness.html
    np.random.seed(seed)
    torch.manual_seed(seed)

  
    if multiple_ads:
        ad_names = ""
        for i, app in enumerate(ad):
            ad_names += '_'
            ad_names += ad[i](["CCC"], ad_features[i]).name
            ad_names += '_'
            ad_names += str(ad_features[i].__name__)
        results_dir = os.path.join(base_results, "lstm_hc", dataset, str(qsar_features.__name__) + ad_names, strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    else:
        results_dir = os.path.join(base_results, "lstm_hc", dataset, str(qsar_features.__name__) + '_' + ad(["CCC"], ad_features).name  + '_' + str(ad_features.__name__), strftime("%Y-%m-%d_%H:%M:%S", gmtime()))
    
    os.makedirs(results_dir, exist_ok=True)
    
    assay_file = f'./datasets/{dataset}.csv'
    df = pd.read_csv(assay_file)
   
    df['features'] = qsar_features(df.smiles)
    df_train, df_test = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=0)
    X1 = np.array(list(df_train['features']))
    X2 = np.array(list(df_test['features']))
    y1 = np.array(list(df_train['label']))
    y2 = np.array(list(df_test['label']))


    # train classifier used for the reward function
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=1, random_state=0)
    clf.fit(X1, y1)
    
    results = {}
    df_train.to_csv(os.path.join(results_dir, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(results_dir, 'test.csv'), index=False)

    scoring_function = ADScoringFunction(clf, qsar_features, ad, ad_features, df_test.smiles, multiple_ads)

    # run optimization
    t0 = time()
    

    smiles_history = optimizer.generate_optimized_molecules(
            scoring_function, 100, starting_population=df_test.smiles, get_history=True)
    
    smiles_history = [[Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles if Chem.MolFromSmiles(s) is not None] for smiles in smiles_history]

    t1 = time()
    opt_time = t1 - t0

    # make a list of dictionaries for every time step
    # this is far from an optimal data structure
    statistics = []
    for optimized_smiles in smiles_history:
        row = {}
        row['smiles'] = optimized_smiles
        row['preds'] = {}
        preds = score(optimized_smiles, clf, scoring_function.featurization, scoring_function.ad, multiple_ads)
        row['preds']['scores'] = preds
        statistics.append(row)

    results['statistics'] = statistics

    stat_time = time() - t1
    results_file = os.path.join(results_dir, 'results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f)
        
    print(f'Storing results in {results_dir}')
    print(f'Optimization time {opt_time:.2f}')
    print(f'Statistics time {stat_time:.2f}')








