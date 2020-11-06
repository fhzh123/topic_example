import os
import pickle
import numpy as np
import pandas as pd

from random import shuffle
from konlpy.tag import Okt
from collections import Counter
from scipy.sparse import dok_matrix
from sklearn.preprocessing import normalize

def preprocessing(args):
    # Data Read
    print('Preprocessing: Data read...')
    data_daelim = pd.read_csv(os.path.join(args.korean_museum_path, '#daelim/daelim_distinct_text_cleansing.csv'))
    data_leeum = pd.read_csv(os.path.join(args.korean_museum_path, '#leeum/leeum_distinct_text_cleansing.csv'))
    data_mmcaseoul = pd.read_csv(os.path.join(args.korean_museum_path, '#mmcaseoul/mmcaseoul_distinct_text_cleansing.csv'))
    data_museumkorea = pd.read_csv(os.path.join(args.korean_museum_path, '#museumkorea/museumkorea_distinct_text_cleansing.csv'))
    data_nfmkorea = pd.read_csv(os.path.join(args.korean_museum_path, '#nfmkorea/nfmkorea_distinct_text_cleansing.csv'))

    # Total Data
    total_text_list = data_daelim['text'].tolist()
    total_text_list.extend(data_leeum['text'].tolist())
    total_text_list.extend(data_mmcaseoul['text'].tolist())
    total_text_list.extend(data_museumkorea['text'].tolist())
    total_text_list.extend(data_nfmkorea['text'].tolist())
    total_text_num = len(total_text_list)

    # Preprocessing
    print('Preprocessing: Term Document Matrix setting...')
    tags = set(['Noun', 'Verb', 'Adjective'])
    stopwords_data = pd.read_csv('./Komoran_stopwords.txt', sep='\t', names=['words',  'tag', 'score'], header=None)
    stopwords = set(stopwords_data['words'])

    print('Daelim start...')
    daelim_results = tdm_make(data_daelim['text'].tolist(), tags, stopwords)
    print('Leeum start...')
    leeum_results = tdm_make(data_leeum['text'].tolist(), tags, stopwords)
    print('MMCA start...')
    mmcaseoul_results = tdm_make(data_mmcaseoul['text'].tolist(), tags, stopwords)
    print('Korea Museum start...')
    museumkorea_results = tdm_make(data_museumkorea['text'].tolist(), tags, stopwords)
    print('NFM start...')
    nfmkorea_results = tdm_make(data_nfmkorea['text'].tolist(), tags, stopwords)
    print('Total start...')
    total_results = tdm_make(total_text_list, tags, stopwords)

    print('Preprocessing: Saving...')
    with open(f'./data/preprocessed.pkl', 'wb') as f:
        pickle.dump({
            'daelim': daelim_results,
            'leeum': leeum_results,
            'mmcaseoul': mmcaseoul_results,
            'museumkorea': museumkorea_results,
            'mfmkorea': nfmkorea_results,
            'total': total_results
        }, f)

def tdm_make(text_list, tags, stopwords):
    okt = Okt()

    word_count = Counter()

    for article in text_list:
        words = list()
        pos = okt.pos(article, norm=True, stem=True)
        if not stopwords == None:
            word_count.update([word for word, tag in pos if tag in tags and len(word) >1 and word not in stopwords])
        else:
            word_count.update([word for word, tag in pos if tag in tags and len(word) >1])            

    wordsList = list()
    raw_text = list()
    index2voca = set()
    for article in text_list:

        words = []
        pos = okt.pos(article, norm=True, stem=True)
        words = [word for word, tag in pos if word_count[word] >= 10]
        if len(words) >= 10:
            index2voca.update(words)
            wordsList.append(words)
            raw_text.append(article)

    index2voca = list(index2voca)
    voca2index = {w: i for i, w in enumerate(index2voca)}

    tdm = dok_matrix((len(wordsList), len(index2voca)), dtype=np.float32)

    for i, words in enumerate(wordsList):
        for word in words:
            tdm[i, voca2index[word]] += 1

    tdm = tdm.tocsr()
    tdm = normalize(tdm)

    return {'index2voca': index2voca, 'voca2index': voca2index, 'wordsList': wordsList, 'tdm': tdm, 'raw_text': raw_text}