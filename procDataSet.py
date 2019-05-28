import os
from itertools import product


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy import sparse


class TrainingQuery(Dataset):
    def __init__(self, raw_path='data', embedded_path='data/embedded'):      
        # read relevance
        raw_training_data = pd.read_csv(os.path.join(raw_path, 'TD.csv'))
        relevances = raw_training_data['Relevance'].values
        self.targets = torch.from_numpy(relevances)
        
        # read embedded data
        self.contents = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_content.npy')))
        self.queries = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_query.npy')))
        
        self.size = self.queries.shape[0]
        
    def __getitem__(self, i):        
        return self.queries[i], self.contents[i], self.targets[i]
    
    def __len__(self):
        return self.size


class TrainingQueryAll(Dataset):
    def __init__(
        self, raw_path='data', embedded_path='data/embedded', 
        is_valid=False, validation_queries=None, normalize01=False, weight=False
    ):      
        # read relevance
        raw_training_data = pd.read_csv(os.path.join(raw_path, 'TD.csv'))
        
        # read embedded data
        self.contents = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_all_content.npy')))
        query_list = np.load(os.path.join(embedded_path, 'ref_training_uniq_query.npy')).tolist()
        query_embedded = np.load(os.path.join(embedded_path, 'encoded_training_uniq_query.npy'))

        # validation split
        if validation_queries is None:
            validation_queries = []
        if is_valid:
            used_queries = validation_queries
        else:
            used_queries = list(set(range(20)) - set(validation_queries))
        used_data = pd.DataFrame()
        for i in used_queries:
            q = query_list[i]
            used_data = used_data.append(raw_training_data[raw_training_data['Query'] == q])
        query_list = [x for i, x in enumerate(query_list) if i in used_queries]
        query_embedded = [x for i, x in enumerate(query_embedded) if i in used_queries]
        self.queries = torch.from_numpy(np.array(query_embedded))

        # data to indices
        self.indices = list(product(range(len(query_list)), range(self.contents.shape[0])))
        self.size = len(self.indices)
        self.targets = sparse.coo_matrix(
            (
                used_data['Relevance'] ,(
                    used_data['Query'].apply(query_list.index), 
                    used_data['News_Index'].apply(lambda x: int(x[5:]))
                )
            ), 
            (len(query_list), self.contents.shape[0])
        ).toarray()            

        self.weight_flag = bool(weight)
#         self.weights = sparse.coo_matrix(
#             (
#                 np.ones(len(used_data)) * (weight-1),(
#                     used_data['Query'].apply(query_list.index), 
#                     used_data['News_Index'].apply(lambda x: int(x[5:]))
#                 )
#             ), 
#             (len(query_list), self.contents.shape[0])
#         ).toarray() + 1
#         self.weights = self.weights / weight
        self.weights = (self.targets * (weight - 1) + 1) / weight
        
        if normalize01:
            self.targets = self.targets / 3
            
    def __getitem__(self, i):     
        j, k = self.indices[i]
        rt = (self.queries[j], self.contents[k], torch.tensor([self.targets[j, k]]))
        if self.weight_flag:
            rt = *rt, torch.tensor([self.weights[j, k]])   
        return rt
    
    def __len__(self):
        return self.size

    
class TrainingQueryAllTwoOut(Dataset):
    def __init__(
        self, raw_path='data', embedded_path='data/embedded', 
        is_valid=False, validation_queries=None
    ):      
        # read relevance
        raw_training_data = pd.read_csv(os.path.join(raw_path, 'TD.csv'))
        
        # read embedded data
        self.contents = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_all_content.npy')))
        query_list = np.load(os.path.join(embedded_path, 'ref_training_uniq_query.npy')).tolist()
        query_embedded = np.load(os.path.join(embedded_path, 'encoded_training_uniq_query.npy'))

        # validation split
        if is_valid:
            used_queries = validation_queries
        else:
            used_queries = list(set(range(20)) - set(validation_queries))
        used_data = pd.DataFrame()
        for i in used_queries:
            q = query_list[i]
            used_data = used_data.append(raw_training_data[raw_training_data['Query'] == q])
        query_list = [x for i, x in enumerate(query_list) if i in used_queries]
        query_embedded = [x for i, x in enumerate(query_embedded) if i in used_queries]
        self.queries = torch.from_numpy(np.array(query_embedded))

        # data to indices
        self.indices = list(product(range(len(query_list)), range(self.contents.shape[0])))
        self.size = len(self.indices)
        
        # targets
        revelances = sparse.coo_matrix(
            (
                used_data['Relevance'] ** 2 + 1 ,(  # 0, 1, 2, 3 -> 1, 2, 5, 10
                    used_data['Query'].apply(query_list.index), 
                    used_data['News_Index'].apply(lambda x: int(x[5:]))
                )
            ), 
            (len(query_list), self.contents.shape[0])
        ).toarray()            
        self.classes = revelances.astype('bool') # 0 or 1
        
        # penality
        revelances[revelances == 1] = -1
        self.revelances = revelances
        
    def __getitem__(self, i):     
        j, k = self.indices[i]
        return self.queries[j], self.contents[k], torch.tensor([self.classes[j, k]]), torch.tensor([self.revelances[j, k]])
    
    def __len__(self):
        return self.size
    

class TestQuery(Dataset):
    def __init__(self, embedded_path='data/embedded'):      
        # read data
        self.contents = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_all_content.npy')))
        self.queries = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_test_query.npy')))

        self.indices = list(product(range(self.queries.shape[0]), range(self.contents.shape[0])))
        self.size = len(self.indices)
        
    def __getitem__(self, i):    
        j, k = self.indices[i]
        return self.queries[j], self.contents[k]
    
    def __len__(self):
        return self.size
