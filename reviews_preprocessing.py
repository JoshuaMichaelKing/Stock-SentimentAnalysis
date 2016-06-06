#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os, sys, codecs, logging, pickle
import jieba
from math import log
import datetime as dt
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
    pos_neg_cut_test()

def pos_neg_cut_test():
    '''
    先通过已构建词典将4月的评论初步分割成积极和消极文本
    '''
    # loading postive and negtive sentiment lexicon
    pos_lexicon_dict = {}
    neg_lexicon_dict = {}
    lexicon = read_lexicon2dict('positive.txt', True)
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)
    lexicon = read_lexicon2dict('hownet-positive.txt')
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)
    lexicon = read_lexicon2dict('ntusd-positive.txt')
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)

    lexicon = read_lexicon2dict('negative.txt', True)
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)
    lexicon = read_lexicon2dict('hownet-negative.txt')
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)
    lexicon = read_lexicon2dict('ntusd-negative.txt')
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)

    review_list_day = ['20160405', '20160406', '20160407', '20160408',
    '20160411', '20160412', '20160413', '20160414', '20160415',
    '2016018', '20160419', '20160420', '20160421',
    '2016025', '20160426', '20160427', '20160429']
    review_list_day = ['20160405']
    pos_reviews = []
    neg_reviews = []
    mid_reviews = []
    print('{0}   {1}'.format(len(review_list_day), review_list_day))

    opentime1 = st.opentime1
    midclose = st.midclose
    opentime2 = st.opentime2
    closetime = st.closetime
    tick_delta = dt.timedelta(minutes=5)

    for subdir in review_list_day:
        tick_now = opentime1
        count = 0
        while True:
            if (tick_now >= opentime1 and tick_now <= midclose) or (tick_now >= opentime2 and tick_now <= closetime):
                hour = tick_now.hour
                minute = tick_now.minute
                fname = str(hour * 100 + minute)
                tick_blog_list = read_file2list(fname, subdir)
                count += len(tick_blog_list)
                for each_blog in tick_blog_list:
                    if each_blog != '':
                        segments = jieba.cut(each_blog)
                        tmp = []
                        for seg in segments:
                            if is_word_invalid(seg) is False:
                                tmp.append(seg)
                        result = sentiment_logarithm_estimation(pos_lexicon_dict, neg_lexicon_dict, tmp)
                        print(result)
                        if result == 0:
                            mid_reviews.append(result)
                        elif result < 0:
                            neg_reviews.append(result)
                        else:
                            pos_reviews.append(result)
                tick_now += tick_delta
            elif tick_now > midclose and tick_now < opentime2:
                tick_now = opentime2
            elif tick_now > closetime:
                break
        print('{0}-{1}'.format(subdir, count))
    print('{0}-{1}-{2}'.format(len(neg_reviews), len(mid_reviews), len(pos_reviews)))

def sentiment_logarithm_estimation(pos_lexicon_dict, neg_lexicon_dict, sentence_blog_segments):
    '''
    using ln((1+pos)/(1+neg)) formula
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

def read_lexicon2dict(fname, isNew=False):
    '''
    read sentiment lexicon to dict
    '''
    lexicon_dict = {}
    filepath = './Dictionary/' + fname
    readfile = codecs.open(filepath, 'r', 'utf-8')
    output = readfile.readlines()  # 对于小文件可以一下全部读出
    for line in output:
        line = line.replace('\n', '')
        if isNew:
            wlist = line.split(' ')
            if len(wlist) > 1:
                lexicon_dict[wlist[0]] = int(wlist[1])
        else:
            line = line.replace(' ', '')
            lexicon_dict[line] = 1
    readfile.close()
    return lexicon_dict

def read_pickle2list(subdir, fname):
    '''
    read pickle to word list and get tfidf
    '''
    filepath = './Data/' + subdir + '/' + fname + '.pkl'
    output = open(filepath, 'rb')
    # Pickle dictionary using protocol 0.
    word_seq = pickle.load(output)
    output.close()
    return word_seq

def save_list2file_pickle(word_list_tfidf, subdir, fname):
    '''
    save word list and tfidf to file by pickle
    '''
    filepath = './Data/' + subdir + '/' + fname + '.pkl'
    output = open(filepath, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(word_list_tfidf, output)
    output.close()

def save_list2file(word_list_tfidf, subdir, fname):
    '''
    save list to txt, add \n to every list member's end
    '''
    filepath = './Data/' + subdir + '/' + fname + '.txt'
    f = codecs.open(filepath, 'a', 'utf-8')
    for tp in word_list_tfidf:
        f.write(tp[0] + ':' + str(tp[1]) + '\n')
    f.close()

def read_file2list(fname, subdir=None):
    '''
    read txt(sina microblog) to list
    '''
    bloglist = []
    filepath = ''
    if subdir is None:
        filepath = './Data/' + fname + '.txt'
    else:
        filepath = './Data/' + subdir + '/' + fname + '.txt'
    readfile = codecs.open(filepath, 'r', 'utf-8')
    output = readfile.readlines()  # 对于小文件可以一下全部读出
    for weibo in output:
        weibo = weibo.replace('\n', '')
        bloglist.append(weibo)
    readfile.close()
    return bloglist

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
