import scipy
import os
import json
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import umap
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit.Chem.Descriptors import qed
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import pickle

import math
from collections import defaultdict

import os.path as op

_fscores = None


def readFragmentScores(name='fpscores'):
    import gzip
    global _fscores
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(os.getcwd(), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateScore(m):
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m,
                                               2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore
def average_tanimoto_similarity(fps_1, fps_2):
    sim = 0
    count = 0
    for fp in fps_1:
        for fp_other in fps_2:
            sim += DataStructs.FingerprintSimilarity(fp, fp_other)
            count += 1
    return sim/count

def max_tanimoto_similarity(fps_1, fps_2):
    sim = 0
    count = 0
    for fp in fps_1:
        for fp_other in fps_2:
            sim = max(sim, DataStructs.FingerprintSimilarity(fp, fp_other))
    return sim

def indexes_identical_fps(fps_1, fps_2):
    indexes = []
    indexes_other = []
    for i, fp in enumerate(fps_1):
        for j, fp_other in enumerate(fps_2):
            if DataStructs.FingerprintSimilarity(fp, fp_other)>0.99:
                indexes.append(i)
                indexes_other.append(j)
    return indexes, indexes_other

def tanimoto_similarities(fps_1, fps_2):
    sim = []
    for fp in fps_1:
        for fp_other in fps_2:
            sim.append(DataStructs.FingerprintSimilarity(fp, fp_other))
    return sim

def ClusterFps(fps,cutoff=0.2):
    # Source : 
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])

    # now cluster the data:
    cs = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
    return cs

def find_cluster(fp, centroids_fp):
    index = 0
    max_sim = 0
    for i, fp_cent in enumerate(centroids_fp):
        sim = DataStructs.FingerprintSimilarity(fp,fp_cent)
        if sim > max_sim:
            max_sim = sim
            index = i
    return index

def return_distribution_cycle_size(smiles):
    max_size = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            ri = m.GetRingInfo()
            n_rings = len(ri.AtomRings())
            max_ring_size = len(max(ri.AtomRings(), key=len, default=()))
            max_size.append(max_ring_size)
    return max_size

def return_distribution_mw(smiles):
    molecular_weights = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            mw = rdMolDescriptors._CalcMolWt(m)
            molecular_weights.append(mw)
    return molecular_weights

def return_distribution_radicals(smiles):
    radicals = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            r = Descriptors.NumRadicalElectrons(m)
            radicals.append(r)
    return radicals

def return_distribution_sulphur(smiles):
    sulphur = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            substructure = Chem.MolFromSmarts('S')
            sulphur.append(len(m.GetSubstructMatches(substructure)))
    return sulphur

def return_distribution_halogen(smiles):
    halogen = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            substructure = Chem.MolFromSmarts('[F,Cl,Br,I]')
            halogen.append(len(m.GetSubstructMatches(substructure)))
    return halogen

def return_distribution_heteroatoms(smiles):
    heteroatoms = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            substructure = Chem.MolFromSmarts('[!C;!H;!c]~[!C;!H;!c]')
            heteroatoms.append(len(m.GetSubstructMatches(substructure)))
    return heteroatoms

def qualitative_analysis(smiles, smiles_test):
    values = []
    distributions = []
    properties = []
    
    max_size = return_distribution_cycle_size(smiles)
    
    values.extend(max_size)
    distributions.extend(["Generated" for _ in range(len(smiles))])
    properties.extend(["Cycle Size" for _ in range(len(smiles))])
    
    max_size_ref = return_distribution_cycle_size(smiles_test)
    values.extend(max_size_ref)
    distributions.extend(["Dataset" for _ in range(len(smiles_test))])
    properties.extend(["Cycle Size" for _ in range(len(smiles_test))])

    molecular_weights = return_distribution_mw(smiles)
    
    values.extend(molecular_weights)
    distributions.extend(["Generated" for _ in range(len(smiles))])
    properties.extend(["Molecular Weights" for _ in range(len(smiles))])
    
    molecular_weights_ref = return_distribution_mw(smiles_test)
    values.extend(molecular_weights_ref)
    distributions.extend(["Dataset" for _ in range(len(smiles_test))])
    properties.extend(["Molecular Weights" for _ in range(len(smiles_test))])

    sulphur = return_distribution_sulphur(smiles)
    values.extend(sulphur)
    distributions.extend(["Generated" for _ in range(len(smiles))])
    properties.extend(["Number of Sulphurs" for _ in range(len(smiles))])
    
    sulphur_ref = return_distribution_sulphur(smiles_test)
    values.extend(sulphur_ref)
    distributions.extend(["Dataset" for _ in range(len(smiles_test))])
    properties.extend(["Number of Sulphurs" for _ in range(len(smiles_test))])
    
    halogen = return_distribution_halogen(smiles)
    values.extend(halogen)
    distributions.extend(["Generated" for _ in range(len(smiles))])
    properties.extend(["Number of Halogens" for _ in range(len(smiles))])
    
    halogen_ref = return_distribution_halogen(smiles_test)
    values.extend(halogen_ref)
    distributions.extend(["Dataset" for _ in range(len(smiles_test))])
    properties.extend(["Number of Halogens" for _ in range(len(smiles_test))])
    

    heteroatoms = return_distribution_heteroatoms(smiles)
    values.extend(heteroatoms)
    distributions.extend(["Generated" for _ in range(len(smiles))])
    properties.extend(["Number of heteroatoms" for _ in range(len(smiles))])
    
    heteroatoms_ref = return_distribution_heteroatoms(smiles_test)
    values.extend(heteroatoms_ref)
    distributions.extend(["Dataset" for _ in range(len(smiles_test))])
    properties.extend(["Number of heteroatoms" for _ in range(len(smiles_test))])

    radicals = return_distribution_radicals(smiles)
    values.extend(radicals)
    distributions.extend(["Generated" for _ in range(len(smiles))])
    properties.extend(["Number of radicals" for _ in range(len(smiles))])
    
    radicals_ref = return_distribution_radicals(smiles_test)
    values.extend(radicals_ref)
    distributions.extend(["Dataset" for _ in range(len(smiles_test))])
    properties.extend(["Number of radicals" for _ in range(len(smiles_test))])
    return values, distributions, properties
    
def quantitative_analysis(smiles, smiles_test, lower_percentile = 0, higher_percentile=100):

    max_size = np.array(return_distribution_cycle_size(smiles))
    max_size_ref = np.array(return_distribution_cycle_size(smiles_test))
    cycle_size_ok = np.mean(np.array(max_size>=np.percentile(max_size_ref, lower_percentile)) & np.array(max_size<=np.percentile(max_size_ref, higher_percentile)))
   
    molecular_weights = return_distribution_mw(smiles)
    molecular_weights_ref = return_distribution_mw(smiles_test)
    molecular_weights_ok = np.mean(np.array(molecular_weights>=np.percentile(molecular_weights_ref, lower_percentile)) & np.array(molecular_weights<=np.percentile(molecular_weights_ref, higher_percentile)))

  
    heteroatoms = return_distribution_heteroatoms(smiles)
    heteroatoms_ref = return_distribution_heteroatoms(smiles_test)
    heteroatoms_ok = np.mean(np.array(heteroatoms>=np.percentile(heteroatoms_ref, lower_percentile))&np.array(heteroatoms<=np.percentile(heteroatoms_ref, higher_percentile)))

    radicals = return_distribution_radicals(smiles)
    radicals_ref = return_distribution_radicals(smiles_test)
    radicals_ok = np.mean(np.array(radicals>=np.percentile(radicals_ref, lower_percentile))&np.array(radicals<=np.percentile(radicals_ref, higher_percentile)))
    
    return 100 * cycle_size_ok, 100 * molecular_weights_ok, 100 * heteroatoms_ok, 100 * radicals_ok

def one_ecfp(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
            m, radius, nBits=1024))
        return fp
    except:
        return None
    
def ecfp4(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_ecfp(s, radius=2) for s in smiles]
    return X

def data_split(dataset):
    """
    Args:
        chid: which assay to use:
        external_file:
    Returns:
        clfs: Dictionary of fitted classifiers
        aucs: Dictionary of AUCs
        balance: Two numbers showing the number of actives in split 1 / split 2
        df1: data in split 1
        df2: data in split 2
    """
    # read data and calculate ecfp fingerprints
    assay_file = f'datasets/{dataset}.csv'
    print(f'Reading data from: {assay_file}')
    df = pd.read_csv(assay_file)
          
    df['ecfp'] = ecfp4(df.smiles)
    df_train, df_test = train_test_split(df, test_size=0.25, stratify=df['label'], random_state=0)
    X1 = np.array(list(df_train['ecfp']))
    X2 = np.array(list(df_test['ecfp']))
    y1 = np.array(list(df_train['label']))
    y2 = np.array(list(df_test['label']))
      
    
    # train classifiers and store them in dictionary
    
    clf = RandomForestClassifier( 
        n_estimators=100, n_jobs=1, random_state=0)
    clf.fit(X1, y1)
    return clf.predict_proba(X2[np.where(y2==1)[0], :])[:, 1], list(df_test.smiles), list(df_test.label), clf
