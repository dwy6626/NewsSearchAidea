from torch.utils.data import Dataset
import pandas as pd


class TrainingQuery(Dataset):
    def __init__(self):      
        # read provided data
        raw_training_data = pd.read_csv('TD.csv')
        news_urls = pd.read_csv('NC_1.csv')
        contents = pd.read_json('url2content.json', typ=pd.Series)
        
        # proccess data
        merged_training = pd.merge(raw_training_data, news_urls, on=['News_Index'])
        
        # construct dataset
        self.queries = merged_training['Query']
        self.contents = contents[merged_training['News_URL']]
        self.targets = merged_training['Relevance']
        
        self.size = len(merged_training)
        
    def __getitem__(self, i):
        return self.queries[i], self.contents[i], self.targets[i]
    
    def __len__(self):
        return self.size
