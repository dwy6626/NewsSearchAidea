from newsDataSet import TrainingQuery
from bert_serving.client import BertClient
from multiprocessing import Pool
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim  as optim
import argparse
from keras.utils import to_categorical

dataset = TrainingQuery("news_data_1")
bc = BertClient(ip='localhost', check_version=False, check_length=False)

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

def pad_sentence(text_token, max_len):
    text_token = [ i for i in text_token if len(i) > 1 and set(i) != set(['\n']) ]
    text_token_len = len(text_token)
    if max_len > text_token_len:
        text_token = text_token  +  ["。"] * ( max_len - text_token_len)
    else:
        text_token = text_token[:max_len]
        
    return text_token

class ARGS():
    lr = 0.01
    batch = 128
    epoch = 10
    num_layers = 1
    seq_len = 30
    word_dim =  768
    hidden_dim = 100

args = ARGS()
# Get model
train_data = DataLoader(dataset, batch_size=args.batch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_Net(args.word_dim, args.hidden_dim, args.num_layers)
model = model.to(device)

training(args, train_data, train_data, model, device,'test.h5')

def training(args, train, valid, model, device, model_name):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
    criterion = nn.CrossEntropyLoss()
    t_batch = len(train) 
    v_batch = len(valid) 

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    
    total_loss, total_acc, best_acc = 0, 0, 0
    train_acc_list = []
    val_acc_list = []
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training set
        for i, (titles, contents ,labels) in enumerate(train):
            titles, contents, labels = list(titles), list(contents), list(labels)
            contents = np.array([ bc.encode(pad_sentence(text.split('。'), 20)) for text in contents])
            titles = bc.encode(titles)
            
            titles = torch.Tensor(titles)
            contents = torch.Tensor(contents)
            labels = torch.LongTensor(labels)
            
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
