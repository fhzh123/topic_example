import pickle
import numpy as np
import pandas as pd
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

    # Wordcloud
    print('Word Cloud Generation: Saving...')
    figure_gen('daelim')
    figure_gen('leeum')
    figure_gen('mmcaseoul')
    figure_gen('museumkorea')
    figure_gen('nfmkorea')
    figure_gen('total')

def figure_gen(museum):
    if museum  == 'daelim':
        dat_ = data_daelim['text']
        mask = np.array(Image.open('./data/Daelim.jpg'))
    elif museum  == 'leeum':
        dat_ = data_leeum['text']
        mask = np.array(Image.open('./data/Leeum.jpg'))
    elif museum  == 'mmcaseoul':
        dat_ = data_mmcaseoul['text']
        mask = np.array(Image.open('./data/mmca.jpg'))
    elif museum  == 'museumkorea':
        dat_ = data_museumkorea['text']
        mask = np.array(Image.open('./data/museumkorea.jpg'))
    elif museum  == 'nfmkorea':
        dat_ = data_nfmkorea['text']
        mask = np.array(Image.open('./data/NFM.jpg'))
    elif museum == 'total':
        dat_ = data_daelim['text'].tolist()
        dat_.extend(data_leeum['text'].tolist())
        dat_.extend(data_mmcaseoul['text'].tolist())
        dat_.extend(data_museumkorea['text'].tolist())
        dat_.extend(data_nfmkorea['text'].tolist())
        dat_ = pd.DataFrame({'text': dat_})['text']
        mask = np.array(Image.open('./data/korea.jpg'))

    wordrank_extractor = KRWordRank(
        min_count = 5,
        max_length = 10,
        verbose = True
        )

    beta = 0.85
    max_iter = 15

    keywords, rank, graph = wordrank_extractor.extract(dat_.tolist(), beta, max_iter)

    keyword_dict = dict()

    for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:300]:
        # print('%8s:\t%.4f' % (word, r))
        keyword_dict[word] = r

    wordcloud = WordCloud(
        font_path = font_path,
        mask=mask,
        width = 800,
        height = 800,
        background_color="white",
    )

    wordcloud = wordcloud.generate_from_frequencies(keyword_dict)
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f'./data/results_{museum}.png', dpi=300)