
from copy import deepcopy

from generate import generate
from featurizers import maccs, ap, ecfp6, ecfp4, ecfp4_counts, fcfp4, physchem, physchem_and_ecfp4, physchem_and_ecfp4_counts, qed
from applicability_domains import filtersvalidity, SMILESvalidity, convex_hull, similarity_max, levenshtein, in_range
from guacamol_baselines.smiles_lstm_hc.smiles_rnn_directed_generator import \
    SmilesRnnDirectedGenerator


# Use same parameters for the SMILES LSTM as in https://github.com/ml-jku/mgenerators-failure-modes/blob/master/run_goal_directed.py

opt_args = {}
opt_args['lstm_hc'] = dict(
    pretrained_model_path='./guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt',
    n_epochs=5,
    mols_to_sample=4112,
    keep_top=128,
    optimize_n_epochs=1,
    max_len=100,
    optimize_batch_size=16,
    number_final_samples=1028,
    sample_final_model_only=False,
    random_start=False,
    smi_file='./data/test.smiles',
    n_jobs=-1,
    canonicalize=False)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import os

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nruns", type = int, help='How many runs to perform per task', default=10)
    parser.add_argument("--base_results", help='Where to store the results', default="results")
    parser.add_argument("--dataset", help='Where to store the results', default="CHEMBL3888429_cleaned")
    parser.add_argument("--n_estimators", type=int, help='Number of trees for the random forest', default=100)
    args = parser.parse_args()
    
    optimizer_args = opt_args['lstm_hc']
    
    # choose dataset: here, requires the file "data/CHEMBL1909140_cleaned.csv"
    for qsar_features in [ecfp4,]:
        for ad in [[in_range, similarity_max]]:
            for ad_features in [[physchem, ecfp6], [physchem, ecfp4], [physchem, ap]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = SmilesRnnDirectedGenerator(**optimizer_args)
                    generate(args.dataset, args.n_estimators, seed, optimizer, 
                             args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[in_range]]:
            for ad_features in [[qed], [ecfp4_counts], [ecfp4], [physchem]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = SmilesRnnDirectedGenerator(**optimizer_args)
                    generate(args.dataset, args.n_estimators, seed, optimizer, 
                             args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[similarity_max]]:
            for ad_features in [[ecfp4], [ap], [ecfp6]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = SmilesRnnDirectedGenerator(**optimizer_args)
                    generate(args.dataset, args.n_estimators, seed, optimizer, 
                            args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[in_range, in_range]]:
            for ad_features in [[physchem, ecfp4], [physchem, ap], [physchem, ecfp6], [physchem, ecfp4_counts]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = SmilesRnnDirectedGenerator(**optimizer_args)
                    generate(args.dataset, args.n_estimators, seed, optimizer, 
                            args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[SMILESvalidity]]: 
            for ad_features in [[ecfp4]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = SmilesRnnDirectedGenerator(**optimizer_args)
                    generate(args.dataset, args.n_estimators, seed, optimizer, 
                             args.base_results, qsar_features, ad, ad_features, True)
