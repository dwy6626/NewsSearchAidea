from bert_serving.client import BertClient
from multiprocessing import Pool
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim  as optim
import argparse
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from torch.functional import F
import os

bc = BertClient(ip='localhost', check_version=False, check_length=False)
class TrainingQuery(Dataset):
    def __init__(self, raw_training_data, news_urls, contents):
        # read provided data
        # proccess data
        merged_training = pd.merge(raw_training_data, news_urls, on=['News_Index'])

        # construct dataset
        self.queries = merged_training['Query']
        self.contents = contents[merged_training['News_URL']]
        self.targets = merged_training['Relevance']
        
        self.contents = list(self.contents)
        self.queries = list(self.queries)
        
        self.contents = np.array([ bc.encode(pad_sentence(text.split('。'), 50)) for text in self.contents])
        self.queries = bc.encode(self.queries)
        
        self.size = len(merged_training)
        
    def __getitem__(self, i):
        return self.queries[i], self.contents[i], self.targets[i]
    
    def __len__(self):
        return self.size

class LSTM_Net(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(LSTM_Net, self).__init__()
        # Fix/Train embedding 
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.title_lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            dropout=dropout,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
        self.content_lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            dropout=dropout,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        
#         self.regressor = nn.Sequential(
#                             nn.Dropout(dropout),
#                             nn.Linear(hidden_dim*2, 1),
#                             nn.SELU())

        self.regressor = nn.Sequential(
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim*2, 4),
                            nn.Softmax())

    def forward(self, titles, contents):
        titles = titles.view(titles.size(0), 1, titles.size(1))
        t, _ = self.title_lstm(titles, None)
        c, _ = self.content_lstm(contents, None)
        t = t[:, -1, :]
        c = c[:, -1, :]
        output = self.regressor(c*t)
        return output
    
def evaluation(outputs, labels):
    values, indices = torch.max(outputs, 1)
    correct = torch.mean(torch.tensor(labels==indices, dtype=torch.float))
#     correct = torch.mean((outputs-labels)**2)
    return correct

def pad_sentence(text_token, max_len):
    text_token = [ i for i in text_token if len(i) > 1 and set(i) != set(['\n']) ]
    text_token_len = len(text_token)
    if max_len > text_token_len:
        text_token = text_token  +  ["。"] * ( max_len - text_token_len)
    else:
        text_token = text_token[:max_len]     
    return text_token

def training(args, train, valid, model, device, model_name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
#     criterion = nn.MSELoss()
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
#             labels = labels.to(device, dtype=torch.float)
            
            optimizer.zero_grad()
            outputs = model(titles, contents)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            
            print('[ Epoch{} == {}/{} ] loss:{:.3f} acc {:.3f}'.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100 / batch_size), end='\r')
            
        print('\nTrain | Loss:{:.5f} acc {:.3f}'.format(total_loss / t_batch, total_acc / t_batch*100))
        
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (titles, contents ,labels) in enumerate(valid):
                titles = titles.to(device)
                contents = contents.to(device)
                labels = labels.to(device)
#                 labels = labels.to(device, dtype=torch.float)
                
                outputs = model(titles, contents)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()
            
            print("Valid | Loss:{:.5f} acc {:.3f}".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                if total_acc / v_batch * 100 > 75:
                    torch.save(model, "{}/mod_{}_ckpt_{:.3f} acc {:.3f}".format('bag_4/', model_name, total_acc/v_batch*100))
                    print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
            
            val_acc_list.append(total_acc / v_batch) 
            
    print(np.min(val_acc_list))

class ARGS():
    lr = 0.0001
    batch = 50
    epoch = 100
    num_layers = 2
    seq_len = 30
    word_dim =  768
    hidden_dim = 200

from sklearn.model_selection import train_test_split
folder = 'news_data_1/'
raw_training_data = pd.read_csv(os.path.join(folder,'TD.csv'))
news_urls = pd.read_csv(os.path.join(folder,'NC_1.csv'))
contents = pd.read_json(os.path.join(folder,'url2content.json'), typ=pd.Series)
train_data, test_data = train_test_split(raw_training_data, test_size=0.1)

train_dataset = TrainingQuery(train_data, news_urls, contents)
test_dataset = TrainingQuery(test_data, news_urls, contents)

args = ARGS()

# Get model
train_data = DataLoader(train_dataset, batch_size=args.batch)
valid_data = DataLoader(test_dataset, batch_size=args.batch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Net(args.word_dim, args.hidden_dim, args.num_layers)
model = model.to(device)

training(args, train_data, valid_data, model, device,'test.h5')

from sklearn.model_selection import train_test_split
folder = 'news_data_1/'
test_query = list(pd.read_csv('./data/QS_1.csv').Query)
test_loader = DataLoader(test_query, batch_size=args.batch, shuffle=False)

batch_size = 100000
test_loader = DataLoader(test_query, batch_size=batch_size, shuffle=False)

class TestQuery(Dataset):
    def __init__(self, contents, test_query):      
        # read data
        self.contents = contents
        self.queries = bc.encode(test_query)
        self.indices = list(product(range(self.queries.shape[0]), range(self.contents.shape[0])))
        self.size = len(self.indices)
        
    def __getitem__(self, i):    
        j, k = self.indices[i]
        return self.queries[j], self.contents[k]
    
    def __len__(self):
        return self.size

    
## test data
from itertools import product
test_data = TestQuery(test_dataset.contents, test_query)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
results = []
model.eval()

with torch.no_grad():
    for q, c in test_loader:
        q, c = q.cuda(), c.cuda()
        outputs = model(q, c)
        results.append(outputs.cpu().data.numpy())   
        
results = np.concatenate(results, axis=0)
results = np.sum(results * np.array([1,2,3,4]), axis=1)

for i in range(4):
    num = np.count_nonzero(np.argmax(results,1) == i)
    print('relevance {}: {}'.format(i,num))

results = results.reshape(20, -1)
search_result = np.flip(np.argsort(results, axis=1), axis=1)

df = pd.DataFrame()
df['Query_Index'] = ['q_{:02d}'.format(i+1) for i in range(20)]

for i in range(300):
    df['Rank_{:03d}'.format(i+1)] = search_result[:, i]

for i, row in df.iterrows():
    df.iloc[i, 1:] = df.iloc[i, 1:].apply(lambda x: 'news_{:06d}'.format(x))
    
fname = 'simple4.csv'
df.to_csv('output/' + fname, index=False)

