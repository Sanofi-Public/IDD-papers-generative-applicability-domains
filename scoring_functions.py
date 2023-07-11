import uuid
from functools import partial
from multiprocessing import Pool
from time import gmtime, strftime

import numpy as np
from guacamol.scoring_function import BatchScoringFunction
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from featurizers import ecfp4
from applicability_domains import convex_hull

def score(smiles_list, clf, featurization, applicability_domain, multiple_ads=False):
    if multiple_ads:
        return score_multiples(smiles_list, clf, featurization, applicability_domain)
    
    """Makes predictions for a list of smiles. Returns none if smiles is invalid"""
    X = featurization(smiles_list)
    AD = applicability_domain.check_smiles_list(smiles_list)
   
    X_valid = [x for x in X if x is not None]
    if len(X_valid) == 0:
        return X

    preds_valid = clf.predict_proba(np.array(X_valid))[:, 1]
    preds = []
    i = 0
    for li, x in enumerate(X):
        if x is None:
            # print(smiles_list[li], Chem.MolFromSmiles(smiles_list[li]), x)
            preds.append(None)
        else:
            preds.append(preds_valid[i]*AD[li])
            assert preds_valid[i] is not None
            i += 1
    return preds


def score_multiples(smiles_list, clf, featurization, applicability_domains):
    """Makes predictions for a list of smiles. Returns none if smiles is invalid"""
    X = featurization(smiles_list)
    ADs = [applicability_domain.check_smiles_list(smiles_list) for applicability_domain in applicability_domains]
    X_valid = [x for x in X if x is not None]
    if len(X_valid) == 0:
        return X
    preds_valid = clf.predict_proba(np.array(X_valid))[:, 1]
    preds = []
    i = 0
    for li, x in enumerate(X):
        if x is None:
            preds.append(None)
        else:
            val = preds_valid[i]
            for ad in ADs:
                val *= ad[li]
            preds.append(val)
            assert preds_valid[i] is not None
            i += 1
    return preds

class ADScoringFunction(BatchScoringFunction):
    def __init__(self, clf, featurization=ecfp4, applicability=convex_hull, applicability_featurization=ecfp4, reference_molecules=None, multiple_ads=False):
        super().__init__()
        self.clf = clf
        self.featurization = featurization
        self.multiple_ads = multiple_ads
        if self.multiple_ads:
            self.ad = [ad(reference_molecules, applicability_featurization[i]) for i, ad in enumerate(applicability)]
        else:
            self.ad = applicability(reference_molecules, applicability_featurization)

    def raw_score_list(self, smiles_list):
        if self.multiple_ads:
            return score_multiples(smiles_list, self.clf, self.featurization, self.ad)
        else:
            return score(smiles_list, self.clf, self.featurization, self.ad)
