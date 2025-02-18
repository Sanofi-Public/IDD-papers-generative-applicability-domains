import uuid
from functools import partial
from multiprocessing import Pool
from time import gmtime, strftime
import json
import csv
import numpy as np
from guacamol.scoring_function import BatchScoringFunction
from rdkit import Chem
from rdkit.Chem import AllChem
from featurizers import ecfp4

class AD():
    """Each applicability domain class inherits from the AD class.
       It must be initialized with a list of SMILES and a type of featurization, and 
       implement a check_smiles_list method, that takes as input a list of SMILES,
       and outputs a list of 0 and 1 according to whether each SMILES correspond to
       a molecule within the AD"""
    def __init__(self, reference_molecules, featurization):
        self.ref = reference_molecules
        self.featurization = featurization

class convex_hull(AD):
    """Check whether a molecule is inside the convex hull of the reference set"""
    def __init__(self, reference_molecules, featurization): 
        super().__init__(reference_molecules, featurization)
        self.table = self.featurization(self.ref)
        variations = []
        values = []
        for i in range(len(np.array(self.table[0]))):
            
            erase = False
            for element in self.table:
                if np.array(element)[i]!=np.array(self.table[0])[i]:
                    erase = True
            if not erase:
                variations.append(1)
                values.append(np.array(element)[i])
            else:
                variations.append(0)
                values.append(0)
        self.variations = np.array(variations)
        self.values = np.array(values)
        self.name = "convex_hull"

    def check_smiles_list(self, smiles_list):
        is_in_convex_hull = []
        for smiles in smiles_list:
            
            X = np.array(self.featurization([smiles])[0])
            
            res = 1 - min(1, np.dot(self.variations, X!=self.values))
            
         
            is_in_convex_hull.append(res)
       
        return is_in_convex_hull

def jaccard(im1, im2):   
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)
    return intersection.sum() / float(union.sum())

class similarity_max(AD):
    def __init__(self, reference_molecules, featurization, threshold=0.5):
        super().__init__(reference_molecules, featurization)
        self.table = self.featurization(self.ref)
        self.threshold = threshold
        self.name = "maxsim"

    def check_smiles_list(self, smiles_list):
        is_in_convex_hull = []
        for smiles in smiles_list:
            if Chem.MolFromSmiles(smiles):
                X = np.array(self.featurization([smiles])[0]) 
                score = max([jaccard(X, y) for y in self.table])
                is_in_convex_hull.append(1 * score>self.threshold)
            else:
                is_in_convex_hull.append(0)
        return is_in_convex_hull

class SMILESvalidity(AD):
    def __init__(self, reference_molecules, featurization):
        super().__init__(reference_molecules, featurization)
        self.name = "smiles_validity"
    def check_smiles_list(self, smiles_list):
        is_in_convex_hull = []
        for smiles in smiles_list:
            if Chem.MolFromSmiles(smiles):
                is_in_convex_hull.append(1)
            else:
                is_in_convex_hull.append(0) 
        return is_in_convex_hull

class filtersvalidity(AD):
    def __init__(self, reference_molecules, featurization):
        super().__init__(reference_molecules, featurization)
        self.name = "filters_validity"
        alert_collection_path = "data/alerts.csv"
        
        names = []
        smarts = []

        with open(alert_collection_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i>0:
                    names.append(row[2])
                    smarts.append(row[3])
        names_already_present = []
        for smiles in reference_molecules: 
            for i, motif in enumerate(smarts):
                subs = Chem.MolFromSmarts(motif)
                if subs != None and Chem.MolFromSmiles(smiles).HasSubstructMatch(subs) and names[i] not in names_already_present:
                    names_already_present.append(names[i])
        self.names = names
        self.smarts = smarts
        self.names_already_present = names_already_present 

            

    def check_smiles_list(self, smiles_list):
        is_in_convex_hull = []
        for smiles in smiles_list:
            if Chem.MolFromSmiles(smiles):
                to_keep = True
                for i, motif in enumerate(self.smarts):
                    subs = Chem.MolFromSmarts(motif)
                    if subs != None and self.names[i] not in self.names_already_present and Chem.MolFromSmiles(smiles).HasSubstructMatch(subs):
                        to_keep = False
                if to_keep:
                    is_in_convex_hull.append(1)
                else:
                    is_in_convex_hull.append(0)
            else:
                is_in_convex_hull.append(0)
        return is_in_convex_hull

class levenshtein(AD):
    def __init__(self, reference_molecules, featurization, threshold=0.8):
        super().__init__(reference_molecules, featurization)
        self.table = self.ref
        self.threshold = threshold
        self.name = "levenshtein"

    def levenshtein_ratio(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        ratio = 1 - distances[-1]/max(len(s1),len(s2))
        return ratio

    def check_smiles_list(self, smiles_list):
        is_in_AD = []
        for smiles in smiles_list:
            if Chem.MolFromSmiles(smiles):
                score = max([levenshtein_ratio(smiles, y) for y in self.table])
                is_in_AD.append(1 * score>self.threshold)
            else:
                is_in_AD.append(0)
        return is_in_AD


class in_range(AD):
    def __init__(self, reference_molecules, featurization, tolerance=0):
        super().__init__(reference_molecules, featurization)
        self.table = self.featurization(self.ref)
        self.mins = []
        self.maxs = []
        
        for i in range(len(np.array(self.table[0]))):
            minimum = 1000
            maximum = -1000
           
            for element in self.table:
               
                if element[i]>maximum:
                    maximum = np.array(element)[i]
                if element[i]<minimum:
                    minimum = np.array(element)[i]
               
            self.mins.append(minimum)
            self.maxs.append(maximum)
           
        self.tolerance = tolerance
        self.name = "range"

    def check_smiles_list(self, smiles_list):
        is_in_AD = []
        for smiles in smiles_list:
            valid = True
            try:
                if Chem.MolFromSmiles(smiles):
                    X = np.array(self.featurization([smiles])[0])
               
                    for i in range(len(X)):
                    
                        if X[i]>self.maxs[i] or X[i]<self.mins[i]:
                            valid = False
                
                    is_in_AD.append(1*valid)
                else:
                    is_in_AD.append(0)

            except:
                is_in_AD.append(0)
        return is_in_AD
