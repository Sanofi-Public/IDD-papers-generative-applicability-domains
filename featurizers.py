import uuid

from functools import partial
from multiprocessing import Pool
from time import gmtime, strftime

import numpy as np
from guacamol.scoring_function import BatchScoringFunction
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from rdkit.Chem import DataStructs, QED
from rdkit.Chem import rdMolDescriptors, FindMolChiralCenters
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def one_ecfp(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
            m, radius, nBits=1024))
        return fp
    except:
        return None

def one_maccs(smile):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(rdkit.Chem.rdMolDescriptors.GetAtomFeatures(m))
        return fp
    except:
        return None

def one_ap(smile):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m))
        return fp
    except:
        return None

def one_ecfp_counts(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:  
        m = Chem.MolFromSmiles(smile)
        fp = AllChem.GetHashedMorganFingerprint(m, 2, nBits=1024)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, array)
       
        
        return array
    except:
        print("Not Working")
        return None 

def one_fcfp(smile, radius=2):
    "Calculate ECFP fingerprint. If smiles is invalid return none"
    try:
        m = Chem.MolFromSmiles(smile)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(
            m, radius, nBits=1024, useFeatures=True))
        return fp
    except:
        return None

def ecfp4(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_ecfp(s, radius=2) for s in smiles]
    return X


def ecfp6(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_ecfp(s, radius=3) for s in smiles]
    return X

def fcfp4(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_fcfp(s, radius=2) for s in smiles]
    return X

def fcfp6(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_fcfp(s, radius=3) for s in smiles]
    return X

def maccs(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_maccs(s) for s in smiles]
    return X

def ecfp4_counts(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_ecfp_counts(s, radius=2) for s in smiles]
    return X

def ap(smiles):
    """Input: list of SMILES
       Output: list of descriptors.
       Compute ECFP4 featurization."""
    X = [one_ap(s) for s in smiles]
    return X

def one_physchem(smile):
    try:
        m = Chem.MolFromSmiles(smile)
        if m is not None:
            hba = rdMolDescriptors.CalcNumHBA(m)
            hbd = rdMolDescriptors.CalcNumHBD(m)
            nrings = rdMolDescriptors.CalcNumRings(m)
            rtb = rdMolDescriptors.CalcNumRotatableBonds(m)
            psa = rdMolDescriptors.CalcTPSA(m)
            logp, mr = rdMolDescriptors.CalcCrippenDescriptors(m)
            mw = rdMolDescriptors._CalcMolWt(m)
            csp3 = rdMolDescriptors.CalcFractionCSP3(m)
            hac = m.GetNumHeavyAtoms()
            
            charges = []
            for at in m.GetAtoms():
                charges.append(at.GetFormalCharge())

            if hac == 0:
                fmf = 0
            else:
                fmf = GetScaffoldForMol(m).GetNumHeavyAtoms() / hac
            ri = m.GetRingInfo()
            n_rings = len(ri.AtomRings())
            max_ring_size = len(max(ri.AtomRings(), key=len, default=()))
            min_ring_size = len(min(ri.AtomRings(), key=len, default=()))
            total_charges = sum(charges)
            min_charge = min(charges)
            max_charge = max(charges)
            n_chiral_centers = len(FindMolChiralCenters(m, includeUnassigned=True))
            return np.array([hba, hbd, hba + hbd, nrings, rtb, psa, logp, mr, mw,
                   csp3, fmf, hac, 
                   max_ring_size, min_ring_size, total_charges, min_charge, max_charge, n_chiral_centers])
    except:
        return None

def physchem(smiles):
    X = [one_physchem(s) for s in smiles]
    return X

def physchem_and_ecfp4(smiles):
    X = [np.concatenate((one_physchem(s), one_ecfp(s, radius=2))) for s in smiles]
    return X

def physchem_and_ecfp4_counts(smiles):
    X = [np.concatenate((one_physchem(s), one_ecfp_counts(s, radius=2))) for s in smiles]
    return X

def qed(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [[QED.qed(mol)] if mol else 0 for mol in mols]
