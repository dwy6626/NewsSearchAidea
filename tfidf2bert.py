import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import train_test_split
from pandas import DataFrame

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--ans_path', type=str)
args = parser.parse_args()


# loading 
raw_training_data = pd.read_csv(os.path.join(args.data_path,'TD.csv')) #Query, News_Index Relevance
news_urls = pd.read_csv(os.path.join(args.data_path,'NC_1.csv')) #News_Index, News_URL
contents = pd.read_json(os.path.join(args.data_path,'url2content.json'), typ=pd.Series) # ['url'] or [i]
query_test = pd.read_csv(os.path.join(args.data_path,'QS_1.csv')) #Query_Index, Query


# Filtering
contents.replace(r'\r',' ',regex=True,inplace=True)
contents.replace(r'\n',' ',regex=True,inplace=True)

stoplist = set(u''':!),.:;?]}¢'"、。〉》」』】〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；%／？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､～￠
々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''')

filterpunt = lambda s: ''.join(filter(lambda x: x not in stoplist, s))

for i in range(contents.shape[0]):
    contents[i] = filterpunt(contents[i])
    
# proccess data
merged_training = pd.merge(raw_training_data, news_urls, on=['News_Index'], how='left')

# construct dataset
queries = merged_training['Query'].values
contents_train = contents[merged_training['News_URL']]
targets = merged_training['Relevance'].values
indices = merged_training['News_Index'].values

Binary = True
if Binary:
    targets = (targets>0).astype('int')

df_bert_dev = pd.DataFrame({'user_id':indices[:230],
                'text_a':contents_train[:230],
                'text_b':queries[:230],
                'label':targets[:230]})

df_bert_train = pd.DataFrame({'user_id':indices[230:],
                'text_a':contents_train[230:],
                'text_b':queries[230:],
                'label':targets[230:]})

df_bert_train = df_bert_train.sample(frac=1)

df_bert_train.to_csv(os.path.join(args.data_path,'train.tsv'), sep='\t', index=False, header=False)
df_bert_dev.to_csv(os.path.join(args.data_path,'dev.tsv'), sep='\t', index=False, header=False)   

#Testing set

candidate = np.loadtxt(args.ans_path, dtype=str, delimiter=',')
candidate = candidate[1:,1:]

query_test_values = query_test['Query'].values
query_test_values = np.repeat(query_test_values, candidate.shape[1])

candidate = candidate.reshape(-1)
index = [int(i[5:])-1 for i in candidate]
candidate_content = contents[index]

df_bert_test = pd.DataFrame({'user_id':['news_'+str(i+1) for i in index],
                'text_a':candidate_content,
                'text_b':query_test_values})

df_bert_test.to_csv(os.path.join(args.data_path, 'test.tsv'), sep='\t', index=False, header=False)