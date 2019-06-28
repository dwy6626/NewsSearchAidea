import multiprocessing

import os
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec



    
# corpus = [
#     [t for t in l if t not in stop_words] for l in sentence_arr.tolist() + [jieba.lcut(q) for q in train_query + test_query.tolist()]
# ]
corpus = np.load('data/corpus.npy')


# word2vec
NumberCPU = multiprocessing.cpu_count()
embedder = Word2Vec(
    corpus,
    size=400, alpha=0.025, window=7, min_count=4, max_vocab_size=None, sample=0.001, 
    workers=NumberCPU, min_alpha=0.0001, 
    sg=1, hs=0, negative=5, cbow_mean=1, 
    iter=0, null_word=0, trim_rule=None, 
    sorted_vocab=1, batch_words=10000
)
print("Dictionary Size: ", len(embedder.wv.vocab.keys()), flush=True)
print("vector length: ", embedder.wv.vector_size, flush=True)



for i in range(100):
    embedder.train(corpus, total_examples=embedder.corpus_count, epochs=1)
    if i % 20 == 9:
        embedder.save(os.path.join('model', 'dim400_iter{}.w2v'.format(i+1)))
    print('epoch', i+1, flush=True)