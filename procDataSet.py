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
    def __init__(self, raw_path='data', embedded_path='data/embedded'):      
        # read relevance
        raw_training_data = pd.read_csv(os.path.join(raw_path, 'TD.csv'))
        relevances = raw_training_data['Relevance'].values
        self.targets = torch.from_numpy(relevances)
        
        # read embedded data
        self.contents = torch.from_numpy(np.load(os.path.join(embedded_path, 'encoded_all_content.npy')))
        query_list = np.load(os.path.join(embedded_path, 'ref_training_uniq_query.npy')).tolist()
        query_embedded = np.load(os.path.join(embedded_path, 'encoded_training_uniq_query.npy'))
        self.queries = torch.from_numpy(query_embedded)
        
        # data to indices
        self.indices = list(product(range(len(query_list)), range(self.contents.shape[0])))
        self.size = len(self.indices)
        self.targets = sparse.coo_matrix(
            (
                raw_training_data['Relevance'] ,(
                    raw_training_data['Query'].apply(query_list.index), 
                    raw_training_data['News_Index'].apply(lambda x: int(x[5:]))
                )
            ), 
            (len(query_list), self.contents.shape[0])
        ).toarray()
        
    def __getitem__(self, i):     
        j, k = self.indices[i]
        return self.queries[j], self.contents[k], torch.tensor(self.targets[j, k])
    
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
