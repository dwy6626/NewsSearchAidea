#!/usr/bin/env python
# coding: utf-8

# In[306]:


from procDataSet import TrainingQuery
from bert_serving.client import BertClient
from multiprocessing import Pool
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim  as optim
import argparse
from keras.utils import to_categorical


# In[323]:


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class TrainingQuery(Dataset):
    def __init__(self, folder='.'):      
        # read provided data
        raw_training_data = pd.read_csv(os.path.join(folder,'TD.csv'))
        news_urls = pd.read_csv(os.path.join(folder,'NC_1.csv'))
        contents = pd.read_json(os.path.join(folder,'url2content.json'), typ=pd.Series)
        # proccess data
        merged_training = pd.merge(raw_training_data, news_urls, on=['News_Index'])

        # construct dataset
        self.queries = merged_training['Query']
        self.contents = contents[merged_training['News_URL']]
        self.targets = merged_training['Relevance']
        self.contents = list(self.contents)
        self.queries = list(self.queries)
        
        self.contents = np.array([ bc.encode(pad_sentence(text.split('。'), 20)) for text in self.contents])
        self.queries = bc.encode(self.queries)
        
        self.size = len(merged_training)
        
    def __getitem__(self, i):
        return self.queries[i], self.contents[i], self.targets[i]
    
    def __len__(self):
        return self.size


# In[324]:


dataset = TrainingQuery("news_data_1")
bc = BertClient(ip='localhost', check_version=False, check_length=False)


# In[350]:


class LSTM_Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(LSTM_Net, self).__init__()
        # Fix/Train embedding 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.title_lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        print(embedding_dim)
        self.content_lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.classifier = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim*2, 4),
                            nn.Softmax())

    def forward(self, titles, contents):
        titles = titles.view(titles.size(0), 1, titles.size(1))
        t, _ = self.title_lstm(titles, None)
        c, _ = self.content_lstm(contents, None)
        t = t[:, -1, :]
        c = c[:, -1, :]
        output = self.classifier(t*c)
        return output
    
def evaluation(outputs, labels):
    values, indices = torch.max(outputs, 1)
    correct = torch.mean(torch.tensor(indices == labels, dtype=torch.float))
    return correct


# In[326]:


def pad_sentence(text_token, max_len):
    text_token = [ i for i in text_token if len(i) > 1 and set(i) != set(['\n']) ]
    text_token_len = len(text_token)
    if max_len > text_token_len:
        text_token = text_token  +  ["。"] * ( max_len - text_token_len)
    else:
        text_token = text_token[:max_len]     
    return text_token


# In[352]:


def training(args, train, valid, model, device, model_name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
    criterion = nn.CrossEntropyLoss()
    t_batch = len(train) 
    v_batch = len(valid) 

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    total_loss, total_acc, best_acc = 0, 0, 0
    train_acc_list = []
    val_acc_list = []
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0      
        # training set
        for i, (titles, contents ,labels) in enumerate(train):      
            titles = titles.to(device)
            contents = contents.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(titles, contents)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            
            print('[ Epoch{} == {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100 / batch_size), end='\r')
            
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} '.format(total_loss / t_batch, total_acc / t_batch*100))


# In[353]:


class ARGS():
    lr = 0.001
    batch = 128
    epoch = 200
    num_layers = 2
    seq_len = 30
    word_dim =  768
    hidden_dim = 100


# In[ ]:


args = ARGS()

# Get model
train_data = DataLoader(dataset, batch_size=args.batch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Net(args.word_dim, args.hidden_dim, args.num_layers)
model = model.to(device)

training(args, train_data, train_data, model, device,'test.h5')

