#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os, sys, codecs, logging
import jieba
import random
from math import log
import datetime as dt

import iohelper
import stocktime as st

reload(sys)
sys.setdefaultencoding('utf-8')

__version__ = '0.0.1'
__license__ = 'MIT'
__author__ = 'Joshua Guo (1992gq@gmail.com)'

'''
Python : Reviews preprocessing.
'''

def main():
    FILE = os.curdir
    logging.basicConfig(filename=os.path.join(FILE,'log.txt'), level=logging.ERROR)
    # pos_neg_cut_test()

    # pos_or_neg_reviews2pkl()    # 将手动标注的积极消极评论集转换为pkl存储
    # pos_tk_lst = iohelper.read_pickle2objects('./Reviews/pos_reviews.pkl') # read training corpus to list[[,...], [,...], ...]
    # neg_tk_lst = iohelper.read_pickle2objects('./Reviews/neg_reviews.pkl') # same as above
    # print('POS_REVIEWS_LENGTH %d\tNEG_REVIEWS_LENGTH %d' % (len(pos_tk_lst), len(neg_tk_lst)))

#-------------------------------------------------------------------------------
def pos_or_neg_reviews2pkl():
    '''
    convert the neg_reviews and pos_reviews to list[[,...], [,...],...]
    save the list to pkl
    '''
    neg_list = iohelper.read_file2list('neg')
    pos_list = iohelper.read_file2list('pos')
    neg_tk_lst = word_tokenization(neg_list)
    pos_tk_lst = word_tokenization(pos_list)    # segmentation : [[,], [,], ...]
    iohelper.save_objects2pickle(neg_tk_lst, './Reviews/neg_reviews.pkl')
    iohelper.save_objects2pickle(pos_tk_lst, './Reviews/pos_reviews.pkl')

def word_tokenization(tick_blog_list):
    '''
    word tokenization by jieba to list
    return list : [[,], [,], ...]
    '''
    count = 0
    seg_list = []
    try:
        for blog in tick_blog_list:
            count += 1
            if blog != '':
                segments = jieba.cut(blog)
                tmp = []
                for seg in segments:
                    tmp.append(seg)
                seg_list.append(tmp)
            else:
                print('Line%d is empty!' % cnt)
    except IOError as e:
        logging.error('IOError %s' % e)
    finally:
        return seg_list

#-------------------------------------------------------------------------------
def pos_neg_cut_test():
    '''
    Based on the initial constructed stock-oriented lexicon, then seperate the whole reviews into pwo part automatically : pos and neg
    '''
    # loading positive and negative sentiment lexicon
    pos_lexicon_dict, neg_lexicon_dict = load_sentiment_lexicon()

    # march + april + may = 40 (40*50=2000, 200 as test 1800 as train)
    date_of_march = ['20160329', '20160331']
    date_of_april = ['20160405', '20160406', '20160407', '20160408',
    '20160411', '20160412', '20160413', '20160414', '20160415',
    '20160418', '20160419', '20160420', '20160421',
    '20160425', '20160426', '20160427', '20160429']
    date_of_may = ['20160503', '20160504', '20160505', '20160506',
    '20160509', '20160510', '20160511', '20160512', '20160513',
    '20160516', '20160517', '20160518', '20160519', '20160520',
    '20160523', '20160524', '20160525', '20160526', '20160527',
    '20160530', '20160531']

    review_list_day = []
    review_list_day.extend(date_of_march)
    review_list_day.extend(date_of_april)
    review_list_day.extend(date_of_may)

    # review_list_day = ['20160329']  # just for test one day : correct pos_reviews and neg_reviews manually
    print('{0}   {1}'.format(len(review_list_day), review_list_day))

    opentime1 = st.opentime1
    midclose = st.midclose
    opentime2 = st.opentime2
    closetime = st.closetime
    tick_delta = dt.timedelta(minutes=5)

    for subdir in review_list_day:
        tick_now = opentime1
        count = 0
        pos_reviews = []
        mid_reviews = []
        neg_reviews = []
        pos_scores = []
        neg_scores = []
        mid_scores = []
        while True:
            if (tick_now >= opentime1 and tick_now <= midclose) or (tick_now >= opentime2 and tick_now <= closetime):
                hour = tick_now.hour
                minute = tick_now.minute
                fname = str(hour * 100 + minute)
                tick_blog_list = iohelper.read_txt2list(fname, subdir)
                # print(tick_blog_list[0])
                # assert 0 == 1
                count += len(tick_blog_list)
                for each_blog in tick_blog_list:
                    if each_blog != '':
                        segments = jieba.cut(each_blog)
                        tmp = []
                        for seg in segments:
                            if is_word_invalid(seg) is False:
                                tmp.append(seg)
                        result = sentiment_logarithm_estimation(pos_lexicon_dict, neg_lexicon_dict, tmp)
                        if result == 0:
                            mid_scores.append(result)
                            mid_reviews.append(each_blog)
                        elif result < 0:
                            neg_scores.append(result)
                            neg_reviews.append(each_blog)
                        else:
                            pos_scores.append(result)
                            pos_reviews.append(each_blog)
                tick_now += tick_delta
            elif tick_now > midclose and tick_now < opentime2:
                tick_now = opentime2
            elif tick_now > closetime:
                break
        print('{0}-{1}'.format(subdir, count))
        mid_reviews = random.sample(mid_reviews, 200)
        iohelper.save_list2file(mid_reviews, './Data/' + subdir + '_mid_reviews')
        print('save_list2file new word[mid polarity] list successfully!')
        neg_reviews = random.sample(neg_reviews, 80)
        pos_reviews = random.sample(pos_reviews, 80)
        iohelper.save_list2file(neg_reviews, './Data/' + subdir + '_neg_reviews')
        iohelper.save_list2file(pos_reviews, './Data/' + subdir + '_pos_reviews')
        print('{0}-{1}-{2}'.format(len(neg_scores), len(mid_scores), len(pos_scores)))

# -----------------------------------------------------------------------------
def load_sentiment_lexicon():
    '''
    载入情感词典
    '''
    # loading positive and negative sentiment lexicon
    pos_lexicon_dict = {}
    neg_lexicon_dict = {}
    lexicon = iohelper.read_lexicon2dict('positive.txt', True)
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('hownet-positive.txt')
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('ntusd-positive.txt')
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)

    lexicon = iohelper.read_lexicon2dict('negative.txt', True)
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('hownet-negative.txt')
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('ntusd-negative.txt')
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)

    return pos_lexicon_dict, neg_lexicon_dict

def sentiment_logarithm_estimation(pos_lexicon_dict, neg_lexicon_dict, sentence_blog_segments):
    '''
    compute every preprocessed sentence's sentiment index
    using ln((1+sigma(pos))/(1+sigma(neg))) formula
    return float : sentiment value
    '''
    pos_list = []
    neg_list = []
    tick_value_tmp = float(0)
    pos_count = 0
    neg_count = 0
    for word in sentence_blog_segments:
        if word in pos_lexicon_dict:
            pos_count += pos_lexicon_dict[word]
        elif word in neg_lexicon_dict:
            neg_count += neg_lexicon_dict[word]
    tick_value_tmp = log(float(1 + pos_count) / float(1 + neg_count))
    return tick_value_tmp

def is_word_invalid(word):
    '''
    to judge the word is or not the chinese, if not return False, else return True.
    '''
    if 'sh' in word or 'sz' in word or 'SH' in word or 'SZ' in word or 'IF' in word or word.isdigit():
        return True
    if word[0] <= chr(127):
        return True  # is english, invalid chinese
    isfloat = True
    try:
        fv = float(word)
    except Exception as e:
        isfloat = False
    return isfloat

if __name__ == '__main__':
    main()
