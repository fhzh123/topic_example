import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from random import shuffle
from konlpy.tag import Mecab
from collections import Counter
from wordcloud import WordCloud
from scipy.sparse import dok_matrix
from krwordrank.word import KRWordRank
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from preprocessing import preprocessing

def main(args):
    if args.preprocessing:
        print('a')
        preprocessing(args)
    
    if args.topic_modeling:
        topic_modeling(args)

    if args.word_cloud:
        word_cloud_gen(args)
    
    else:
        print('Choose task in [preprocessing, topic_modeling, word_cloud_gen]')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='args Parser')
    parser.add_argument('--preprocessing', action='store_true')
    parser.add_argument('--topic_modeling', action='store_true')
    parser.add_argument('--word_cloud', action='store_true')
    parser.add_argument('--korean_museum_path', default='./data/GCPKoreanMuseumData', type=str)
    parser.add_argument('--font_path', default='/usr/share/fonts/truetype/nanum/NanumGothic.ttf', type=str)
    parser.add_argument('--K', default=10, type=int)
    args = parser.parse_args()
    main(args)