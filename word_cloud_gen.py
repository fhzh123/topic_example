import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from wordcloud import WordCloud
from krwordrank.word import KRWordRank

def word_cloud_gen(args):
    # Data Read
    print('Word Cloud Generation: Data read...')
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

    # Wordcloud
    print('Word Cloud Generation: Saving...')
    figure_gen(data_daelim['text'].tolist(), np.array(Image.open('./data/Daelim.jpg')), 'daelim', args)
    figure_gen(data_leeum['text'].tolist(), np.array(Image.open('./data/Leeum.jpg')), 'leeum', args)
    figure_gen(data_mmcaseoul['text'].tolist(), np.array(Image.open('./data/mmca.jpg')), 'mmcaseoul', args)
    figure_gen(data_museumkorea['text'].tolist(), np.array(Image.open('./data/museumkorea.jpg')), 'museumkorea', args)
    figure_gen(data_nfmkorea['text'].tolist(), np.array(Image.open('./data/NFM.jpg')), 'nfmkorea', args)
    figure_gen(total_text_list, np.array(Image.open('./data/korea.jpg')), 'total', args)

def figure_gen(dat_, mask, save_name, args):
    wordrank_extractor = KRWordRank(
        min_count = 5,
        max_length = 10,
        verbose = True
        )

    beta = 0.85
    max_iter = 10

    keywords, rank, graph = wordrank_extractor.extract(dat_, beta, max_iter)

    keyword_dict = dict()

    for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:args.word_cloud_words]:
        # print('%8s:\t%.4f' % (word, r))
        keyword_dict[word] = r

    wordcloud = WordCloud(
        font_path = args.font_path,
        mask=mask,
        width = 800,
        height = 800,
        background_color="white",
    )

    wordcloud = wordcloud.generate_from_frequencies(keyword_dict)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f'./data/results_{save_name}.png', dpi=300)
    plt.show()