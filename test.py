from multiprocessing import Pool
from torch.utils.data import TensorDataset, DataLoader, Dataset

import torch
import torch.nn as nn
import torch.optim  as optim
import argparse
import pandas as pd
import numpy as np
from torch.functional import F

import os
import jieba
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output
from itertools import product
from keras.utils import to_categorical

## setting the jieba
import jieba.posseg as pseg
import numpy as np

import multiprocessing
from gensim.summarization import bm25

NumberCPU = multiprocessing.cpu_count()
jieba.initialize()

def jbcut(x):
    if x is not None:
        sen = jieba.lcut(x, cut_all=False)
        sen = [i for i in sen if i not in stop_words]
        return sen
    else:
        return None
    
def histogram_equalization(image, number_bins=256):
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 100 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized

jieba.load_userdict(os.path.join('data', 'dict.txt.big'))
[jieba.add_word(i, freq=None, tag=None) for i in ['不支持','文林苑', '都更案','十八趴','證所稅','前瞻建設']]

top_num = 300

with open('data/stop_word_zhtw.txt') as file:
    data = file.read()
stop_words = data.split('\n')
stop_words += ['「', '」', '，', '\n', '）', '（', ')', '(']



folder = 'data/'
raw_training_data = pd.read_csv(os.path.join(folder,'TD.csv'))
news_urls = pd.read_csv(os.path.join(folder,'NC_1.csv'))
contents = pd.read_json(os.path.join(folder,'url2content.json'), typ=pd.Series)
test_query = np.array(pd.read_csv('./data/QS_1.csv').Query)
## sort the contents by index
keys, content_list = contents.keys(), contents.values
    
pool = multiprocessing.Pool(processes=NumberCPU)
sentence_arr = pool.map(jbcut,content_list)
pool.close()
pool.join()


## initialize tfidf model
bm25Model = bm25.BM25(sentence_arr)

train_query = list(set(raw_training_data.Query))

y_train = []
y_index = {}
for i  in train_query:
    index = np.where(raw_training_data.Query==i)
    data = raw_training_data.iloc[index]
    y = dict(zip(data.News_Index,data.Relevance))
    y_idx = [ (int(idx.split('_')[1])-1, rel )  for idx, rel in zip(data.News_Index, data.Relevance)]
    y_train.append(y)
    y_index[i] = [y_idx]

common_query = set(train_query) & set(test_query)
common_query = list(common_query)
common = [ train_query.index(i) for i in common_query]

total_scores = list()
for test_id , text_q in enumerate(test_query):
    text = jieba.lcut(text_q)
    text = [ t for t in text if t not in stop_words]
    
    scores = np.array(bm25Model.get_scores(text))
    scores = histogram_equalization(scores)
    
    top_query = np.argsort(scores)[::-1][:5]
    irrelevant_query = np.argsort(scores)[:5]
    
    print(test_id ,text_q, text)
    print("top query num {}".format(len(top_query)))
    
    for query in top_query:
        all_words = [ (sentence_arr[query][cnt], bm25Model.get_score(sentence_arr[query], cnt))
                     for cnt, i in enumerate(sentence_arr[query])]

        all_words = sorted(all_words,key=lambda x:(x[1]))[::-1]
        top_words = all_words[:200]
        top_words = [x[0] for x in top_words]
        text += top_words
    
    expansion_score = np.array(bm25Model.get_scores(text))
    expansion_score = histogram_equalization(expansion_score)
    scores += 7 * expansion_score
    
    top_num = 300
    keys = pd.DataFrame(np.argsort(scores)[::-1][:top_num])
    ans = keys[0].apply(lambda x: 'news_{:06d}'.format(x+1))
    
    delta = np.max(scores) 
    if text_q in y_index.keys():
        for idx, rel in (y_index[text_q][0]):
            scores[idx] += delta * rel
    
    total_scores.append(scores)

search_result = np.zeros((20,top_num))
for cnt,i in enumerate(total_scores):
    keys = np.argsort(i)[::-1][:top_num]
    search_result[cnt] += keys
    
search_result = search_result.astype(np.int)

df = pd.DataFrame()
df['Query_Index'] = ['q_{:02d}'.format(i+1) for i in range(20)]

for i in range(top_num):
    df['Rank_{:03d}'.format(i+1)] = search_result[:, i] + 1

for i, row in df.iterrows():
    df.iloc[i, 1:] = df.iloc[i, 1:].apply(lambda x: 'news_{:06d}'.format(x))

fname = 'final'
df.to_csv('output/' + fname, index=False)