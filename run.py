import os
from copy import deepcopy

from generate import generate
from featurizers import maccs, ap, ecfp6, ecfp4, ecfp4_counts, fcfp4, physchem, physchem_and_ecfp4, physchem_and_ecfp4_counts, qed
from applicability_domains import filtersvalidity, SMILESvalidity, convex_hull, similarity_max, levenshtein, in_range
from guacamol_baselines.smiles_lstm_hc.smiles_rnn_directed_generator import \
    SmilesRnnDirectedGenerator
from guacamol_baselines.graph_ga.goal_directed_generation import GB_GA_Generator
from guacamol_baselines.smiles_ga.goal_directed_generation import ChemGEGenerator
from guacamol_baselines.graph_mcts.goal_directed_generation import GB_MCTS_Generator

# Use same parameters for the SMILES LSTM as in https://github.com/ml-jku/mgenerators-failure-modes/blob/master/run_goal_directed.py

opt_args = {}
opt_args['lstm_hc'] = dict(
    pretrained_model_path='./guacamol_baselines/smiles_lstm_hc/pretrained_model/model_final_0.473.pt',
    n_epochs=151,
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

opt_args['graph_ga'] = dict(
    smi_file = 'data/utils.smi', 
    population_size=100,
    offspring_size=200,
    mutation_rate=0.01,
    generations=1000,
    n_jobs=-1,
    random_start=False,
    patience=5)

opt_args['smiles_ga'] = dict(
    smi_file = 'data/utils.smi',
    n_mutations=200,
    gene_size=300,
    population_size=100,
    generations=1000,
    n_jobs=-1,
    random_start=False,
    patience=5)

opt_args['graph_mcts'] = dict(
    pickle_directory = '/common/workdir/AD4AI/guacamol_baselines/graph_mcts',
    generations=1000,
    population_size=100,
    num_sims=40,
    max_children=25,
    max_atoms=60,
    init_smiles='CC',
    n_jobs=-1,
    patience=5)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    import os

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nruns", type = int, help='How many runs to perform per task', default=10)
    parser.add_argument("--base_results", help='Where to store the results', default="results")
    parser.add_argument("--dataset", help='Where to store the results', default="CHEMBL3888429_cleaned")
    parser.add_argument("--n_estimators", type=int, help='Number of trees for the random forest', default=100)
    parser.add_argument("--generator", type=str, help='Generator used', default="lstm_hc")

    args = parser.parse_args()
    
    optimizer_args = opt_args[args.generator]
    
    if args.generator == 'graph_mcts' and optimizer_args['pickle_directory'] is None:
        optimizer_args['pickle_directory'] = os.path.dirname(os.path.realpath(__file__))
        
    optimizers = {"graph_ga": GB_GA_Generator, "lstm_hc": SmilesRnnDirectedGenerator, "smiles_ga": ChemGEGenerator, "graph_mcts": GB_MCTS_Generator}
    
    # choose dataset: here, requires the file "data/CHEMBL1909140_cleaned.csv"
    for qsar_features in [ecfp4,]:
        for ad in [[in_range, similarity_max]]:
            for ad_features in [[physchem, ecfp6], [physchem, ecfp4], [physchem, ap]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = optimizers[args.generator](**optimizer_args)
                    generate(args.generator, args.dataset, args.n_estimators, seed, optimizer, 
                             args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[in_range]]:
            for ad_features in [[qed], [ecfp4_counts], [ecfp4], [physchem]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = optimizers[args.generator](**optimizer_args)
                    generate(args.generator, args.dataset, args.n_estimators, seed, optimizer, 
                             args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[similarity_max]]:
            for ad_features in [[ecfp4], [ap], [ecfp6]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = optimizers[args.generator](**optimizer_args)
                    generate(args.generator, args.dataset, args.n_estimators, seed, optimizer, 
                            args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[in_range, in_range]]:
            for ad_features in [[physchem, ecfp4], [physchem, ap], [physchem, ecfp6], [physchem, ecfp4_counts]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = optimizers[args.generator](**optimizer_args)
                    generate(args.generator, args.dataset, args.n_estimators, seed, optimizer, 
                            args.base_results, qsar_features, ad, ad_features, True)

    for qsar_features in [ecfp4]:
        for ad in [[SMILESvalidity]]: 
            for ad_features in [[ecfp4]]:
                for i in range(0, args.nruns):
                    seed = i 
                    optimizer = optimizers[args.generator](**optimizer_args)
                    generate(args.generator, args.dataset, args.n_estimators, seed, optimizer, 
                             args.base_results, qsar_features, ad, ad_features, True)
