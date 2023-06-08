
import spacy
import numpy as np
import jieba
import pandas as pd
import json
from gensim.models import Word2Vec
from tqdm import tqdm
from typing import Union, List, Dict, Tuple
from pathlib import Path
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TCSP import read_stopwords_list


# Get the stopwords
stopwords = read_stopwords_list()

def tokenize(text: str, stopwords: list) -> str:
    tokens = jieba.cut(text.replace(' ',''))
    return " ".join([w for w in tokens if w not in stopwords])


temp_list = []
train = pd.read_json("dataset/ptt_tsmc_2019.json")
for word in train['comments']:
    temp_comments = word.split("'")[:-15]
    a = [x[:-15] for x in temp_comments]
    string = ' '.join(a)
    temp_list.append(string)
train['new_comments'] = temp_list    
train['process'] = train.new_comments.apply(partial(tokenize, stopwords=stopwords))
train.drop(['comments','new_comments'],axis=1,inplace=True)
print(train)
# train.to_csv('./dataset/2330_word4.json',index=False,encoding='utf-8')
# print(train)

full_train = pd.read_csv('./dataset/2330_word4.json',encoding='utf-8')

def w2w(sentence):
    result = sentence.split(" ")
    return result

def w2v(word,model):
    word  = word
    model = model
    vector = model.wv[word]
    ave_vector = np.mean(vector,axis = 0)
    return ave_vector.tolist()

full_train_list = full_train.process.apply(w2w).to_list()

print(full_train_list)

model = Word2Vec(full_train_list, vector_size=64, window=5, min_count=1)

vec_list = [w2v(x,model) for x in full_train_list]

print(len(vec_list))
print(vec_list)

np.savetxt('dataset\w2v.txt',np.array(vec_list).reshape(-1, 1))
