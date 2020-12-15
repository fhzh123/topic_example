import os
import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import NMF, LatentDirichletAllocation

def topic_modeling(args):
    # Data read
    print('Topic Modeling: Data read...')
    with open('./data/preprocessed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print('Daelim start...')
    daelim_results = nmf_topic_modeling(data['daelim'], args.K)
    print('Leeum start...')
    leeum_results = nmf_topic_modeling(data['leeum'], args.K)
    print('MMCA start...')
    mmcaseoul_results = nmf_topic_modeling(data['mmcaseoul'], args.K)
    print('Korea Museum start...')
    museumkorea_results = nmf_topic_modeling(data['museumkorea'], args.K)
    print('NFM start...')
    nfmkorea_results = nmf_topic_modeling(data['mfmkorea'], args.K)
    print('Total start...')
    total_results = nmf_topic_modeling(data['total'], args.K)

    print('Topic Modeling: Saving...')
    with open('./data/topic_results.pkl', 'wb') as f:
        pickle.dump({
            'daelim': daelim_results,
            'leeum': leeum_results,
            'mmcaseoul': mmcaseoul_results,
            'museumkorea': museumkorea_results,
            'mfmkorea': nfmkorea_results,
            'total': total_results
        }, f)

def nmf_topic_modeling(dat_, K):
    topic_list = list()
    nmf = LatentDirichletAllocation(n_components=K, max_iter=1000)
    W = nmf.fit_transform(dat_['tdm'])
    H = nmf.components_

    for k in range(K):
        print(f"{k}th topic")
        pre_topic_list = list()
        for index in H[k].argsort()[::-1][:20]:
            print(dat_['index2voca'][index], end=" ")
            pre_topic_list.append(dat_['index2voca'][index])
        print("\n")
        topic_list.append(pre_topic_list)

    return topic_list