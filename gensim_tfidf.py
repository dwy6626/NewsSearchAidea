import jieba
import pandas as pd
import numpy as np
import os
from pandas import DataFrame
from multiprocessing import Pool
jieba.load_userdict(os.path.join('data', 'dict.txt.big'))

def tokenize(sentence):
    """ Use jieba to tokenize a sentence.
    Args:
        sentence (str): One string.
    Return: 
        tokens (list of str): List of tokens in a sentence.
    """
    words = jieba.cut(sentence)
    tokens = []
    for word in words:
        tokens.append(word)
    return tokens

# loading 
raw_training_data = pd.read_csv(os.path.join('data','TD.csv')) #Query, News_Index Relevance
news_urls = pd.read_csv(os.path.join('data','NC_1.csv')) #News_Index, News_URL
contents = pd.read_json(os.path.join('data','url2content.json'), typ=pd.Series) # ['url'] or [i]
query_test = pd.read_csv(os.path.join('data','QS_1.csv')) #Query_Index, Query

contents.replace(r'\r',' ',regex=True,inplace=True)
contents.replace(r'\n',' ',regex=True,inplace=True)

print('Starting tokenize...')
P = Pool(processes=4) 
data = P.map(tokenize, contents) #[['新北市', '第二', ...], [...], ...]
P.close()
P.join()

print('Building dictionary...')
from gensim import corpora
dictionary = corpora.Dictionary(data)
corpus = [dictionary.doc2bow(text) for text in data]
corpora.MmCorpus.serialize('corpus.mm', corpus) 

from gensim import models
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

corpora.MmCorpus.serialize('corpus.mm', corpus_tfidf) 
tfidf.save("my_model.tfidf")
tfidf = models.TfidfModel.load("my_model.tfidf")

print('Building LsiModel...')
corpus_tfidf = corpora.MmCorpus('corpus.mm')
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)

print('Building MatrixSimilarity...')
from gensim.similarities import MatrixSimilarity
index = MatrixSimilarity(lsi[corpus_tfidf])

index.save('deerwester.index')
index = MatrixSimilarity.load('deerwester.index')

print('Testing...')
result = np.zeros((20,300)).astype('str')
j = 0
for doc in query_test['Query']:
    doc = jieba.cut(doc)
    tokens = []
    for word in doc:
        tokens.append(word)

    vec_bow = dictionary.doc2bow(tokens)
    vec_lsi = lsi[vec_bow]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for i in range(300):
        result[j][i] = 'news_' + str(sims[i][0]+1).zfill(6)
    
    j += 1

left = np.arange(1,21).astype('str')
left = np.core.defchararray.zfill(left,2)
left = np.core.defchararray.add('q_',left)

result = np.concatenate((left.reshape(-1,1),result),axis=1)

title = np.asarray(['Rank_'+str(i).zfill(3) for i in range(1,301)])
Query_Index = np.array(['Query_Index'])
title = np.concatenate((Query_Index,title)).reshape(1,-1)

ans = np.concatenate((title,result),axis=0)

np.savetxt('ans.csv', ans, delimiter=",",fmt='%s')





