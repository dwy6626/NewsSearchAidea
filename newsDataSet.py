import re


import torch
from torch.utils.data import Dataset
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


class TrainingQuery(Dataset):
    def __init__(self):      
        # read provided data
        raw_training_data = pd.read_csv('TD.csv')
        news_urls = pd.read_csv('NC_1.csv')
        contents = pd.read_json('url2content.json', typ=pd.Series)
        
        # proccess data
        merged_training = pd.merge(raw_training_data, news_urls, on=['News_Index'])
        contents = contents.apply(lambda x: ' [CLS] ' + re.sub('[\n，。、]+', ' [SEP] ', x))
        
        # construct dataset
        self.queries = merged_training['Query']
        self.contents = contents[merged_training['News_URL']]
        self.targets = merged_training['Relevance']
        
        self.size = len(merged_training)
        
    def __getitem__(self, i):
        
        padding_size = 512
        tokenized_text = tokenizer.tokenize(self.contents[i])

        if len(tokenized_text) > padding_size:
            tokenized_text = tokenized_text[:padding_size]
        else:
            tokenized_text = tokenized_text + ['[PAD]'] * (padding_size - len(tokenized_text))

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_tensors = torch.zeros(padding_size).long()
        tokens_tensor = torch.tensor(indexed_tokens)
        
        return self.queries[i], tokens_tensor, segments_tensors, self.targets[i]
    
    def __len__(self):
        return self.size
