import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import os
import numpy as np
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
bc = BertClient(check_length=False)

raw_training_data = pd.read_csv(os.path.join('data','TD.csv'))
news_urls = pd.read_csv(os.path.join('data','NC_1.csv'))
contents = pd.read_json(os.path.join('data','url2content.json'), typ=pd.Series)
query_test = pd.read_csv(os.path.join('data','QS_1.csv'))
# proccess data
merged_training = pd.merge(raw_training_data, news_urls, on=['News_Index'])

# construct dataset
queries = query_test['Query'].values
contents_byurl = contents[merged_training['News_URL']]
targets = merged_training['Relevance'].values
indices = raw_training_data['News_Index'].values

# encode query
queries_dic = {}
for query in queries:
    if query not in queries_dic:
        queries_dic[query] = bc.encode([query])

em_query = np.zeros((20, 768))
query_list = []
i = 0
for query in queries_dic.keys():
    em_query[i] = queries_dic[query]
    query_list.append(query)
    i += 1

#encode contents
em_news = np.zeros((contents.shape[0], 768))
for i in range(em_news.shape[0]):
    if i%100==0:
        print(i)
    if contents[i]=='' or contents[i]==' ':
        contents[i] = '沒資料'
    
    em_news[i] = bc.encode([contents[i]])

np.save('em_news.npy',em_news)
em_news = np.load('em_news.npy')

similarity = cosine_similarity(em_news, em_query)

sort_index = np.argsort(-similarity, axis=0)
sort_index = sort_index[:300].transpose()
sort_index = np.core.defchararray.zfill((sort_index+1).astype('str'),6)
sort_index = np.core.defchararray.add('news_',sort_index)

left = np.arange(1,21).astype('str')
left = np.core.defchararray.zfill(left,2)
left = np.core.defchararray.add('q_',left)

sort_index = np.concatenate((left.reshape(-1,1),sort_index),axis=1)

title = np.asarray(['Rank_'+str(i).zfill(3) for i in range(1,301)])
Query_Index = np.array(['Query_Index'])
title = np.concatenate((Query_Index,title)).reshape(1,-1)

ans = np.concatenate((title,sort_index),axis=0)

np.savetxt('ans.csv', ans, delimiter=",",fmt='%s')