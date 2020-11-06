import os
import argparse

from preprocessing import preprocessing
from topic_modeling import topic_modeling
from word_cloud_gen import word_cloud_gen

def main(args):
    if args.preprocessing:
        print('Preprocessing...')
        preprocessing(args)
    
    if args.topic_modeling:
        print('Topic Modeling...')
        topic_modeling(args)

    if args.word_cloud:
        print('Wordcloud Generation...')
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
    parser.add_argument('--word_cloud_words', default=300, type=int)
    args = parser.parse_args()
    main(args)