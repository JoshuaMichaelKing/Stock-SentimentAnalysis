#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os, sys, codecs, logging
import jieba
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
Python : Using Sentiment-Lexicon to Implement Feature Selection and Sentiment Index Computing.
'''

def main():
    FILE = os.curdir
    logging.basicConfig(filename=os.path.join(FILE, 'log.txt'), level=logging.ERROR)

    # loading postive and negtive sentiment lexicon
    pos_lexicon_dict = {}
    neg_lexicon_dict = {}
    lexicon = iohelper.read_lexicon2dict('hownet-positive.txt')
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('ntusd-positive.txt')
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('positive.txt', True)
    pos_lexicon_dict = dict(pos_lexicon_dict, **lexicon)

    lexicon = iohelper.read_lexicon2dict('hownet-negative.txt')
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('ntusd-negative.txt')
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)
    lexicon = iohelper.read_lexicon2dict('negative.txt', True)
    neg_lexicon_dict = dict(neg_lexicon_dict, **lexicon)

    print('pos_lexicon_dict length : %d' % len(pos_lexicon_dict))
    print('neg_lexicon_dict length : %d' % len(neg_lexicon_dict))

    print('---------------Sentiment Lexicon------------------')
    sentiment_lexicon_compute(pos_lexicon_dict, neg_lexicon_dict)

# -----------------------------------------------------------------------------
def sentiment_lexicon_compute(pos_lexicon_dict, neg_lexicon_dict):
    '''
    1. select the words to construct dictionary
    2. compute sentiment index according to the stock-oriented dictionary
    '''

    opentime1 = st.opentime1
    midclose = st.midclose
    opentime2 = st.opentime2
    closetime = st.closetime
    tick_delta = dt.timedelta(minutes=5)

    status = raw_input("Construct dictionary or compute sentiment index? Please input yes(lexicon construction) or no(compute sentiment index)!")
    isPrint = False  # use the flag to print the word score in sentiment computing
    log_or_not = raw_input("Using logarithm(yes) or not(no)? Please input yes or no!")

    review_list_day = []
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

    # june = 10 (compute 10 days sentiment index to make correlation analysis)
    date_of_june = ['20160606',
    '20160613', '20160614', '20160615',
    '20160620', '20160622', '20160624', '20160628']
    review_list_day.extend(date_of_march)
    review_list_day.extend(date_of_april)
    review_list_day.extend(date_of_may)
    review_list_day.extend(date_of_june) # just for test by month
    # review_list_day = ['20160601']

    for subdir in review_list_day:
        tick_now = opentime1
        blog_corpus = []
        sentiment_index = []
        print('>>>The date to be handled : {0}'.format(subdir))
        while True:
            if (tick_now >= opentime1 and tick_now <= midclose) or (tick_now >= opentime2 and tick_now <= closetime):
                hour = tick_now.hour
                minute = tick_now.minute
                if hour is 9 and minute is 35:
                    isPrint = True
                else:
                    isPrint = False
                fname = str(hour * 100 + minute)
                blog_five_min_list = iohelper.read_txt2list(fname, subdir)
                tick_blog_list = word_tokenization(blog_five_min_list)   # convert 5-min reviews to blog list lisk this : [[,], [,],...]
                blog_corpus.extend(tick_blog_list)
                tick_now += tick_delta
                # Compute Sentiment Index
                if status != 'yes':
                    score_tmp = 0
                    if log_or_not == 'yes':
                        score_tmp = sentiment_compute_logarithm(pos_lexicon_dict, neg_lexicon_dict, tick_blog_list, isPrint)
                    else:
                        score_tmp = sentiment_compute_average(pos_lexicon_dict, neg_lexicon_dict, tick_blog_list, isPrint)
                    sentiment_index.append(score_tmp)
            elif tick_now > midclose and tick_now < opentime2:
                tick_now = opentime2
            elif tick_now > closetime:
                break
        # not necessary if you have processed it to word_tfidf list txt pkl
        if status == 'yes':
            word_preprocessing(blog_corpus, subdir)
            print('%s : word selected from blog_corpus successfully!' % (subdir))
        else:
            iohelper.save_list2pickle(sentiment_index, subdir, 'saindex_seq')
            print('%s : save_list2pickle successfully! %d' % (subdir, len(sentiment_index)))

    print('Ending.....')

# -----------------------------------------------------------------------------
def sentiment_compute_average(pos_lexicon_dict, neg_lexicon_dict, tick_blog_segments, isPrint):
    '''
    <<<obsolete method>>>
    basic plus for positive and minus for negative to compute index average
    '''
    index_list = []
    tick_value_tmp = float(0)
    for doc in tick_blog_segments:
        sentence_count = 0
        doc_tmp = []
        for word in doc:
            if is_word_invalid(word):
                continue
            else:
                doc_tmp.append(word)
        for word in doc_tmp:
            if word in pos_lexicon_dict:
                if isPrint:
                    print("%s + %d" % (word, pos_lexicon_dict[word]))
                sentence_count += pos_lexicon_dict[word]
            elif word in neg_lexicon_dict:
                if isPrint:
                    print("%s - %d" % (word, neg_lexicon_dict[word]))
                sentence_count -= neg_lexicon_dict[word]
        index_list.append(sentence_count)
    if len(index_list) is not 0:
        tick_value_tmp = sum(index_list) / float(len(index_list))
    # if isPrint:
    print('average : %f' % tick_value_tmp)
    return tick_value_tmp

# -----------------------------------------------------------------------------
def sentiment_compute_logarithm(pos_lexicon_dict, neg_lexicon_dict, tick_blog_segments, isPrint = False):
    '''
    using ln((1+sigma[pos])/(1+sigma[neg])) formula
    '''
    pos_list = []
    neg_list = []
    tick_value_tmp = float(0)
    for doc in tick_blog_segments:
        pos_count = 0
        neg_count = 0
        doc_tmp = []
        for word in doc:
            if is_word_invalid(word):
                continue
            else:
                doc_tmp.append(word)
        for word in doc_tmp:
            if word in pos_lexicon_dict:
                pos_count += pos_lexicon_dict[word]
            elif word in neg_lexicon_dict:
                neg_count += neg_lexicon_dict[word]
        pos_list.append(pos_count)
        neg_list.append(neg_count)
    tick_value_tmp = log(float(1 + sum(pos_list)) / float(1 + sum(neg_list)))
    if isPrint:
        print('FIRST-5-MIN Index : %f' % tick_value_tmp)
    return tick_value_tmp

# -----------------------------------------------------------------------------
def word_preprocessing(blog_corpus, subdir):
    '''
    blog_corpus : [["", ""], ["", ""], ["", ""]], new_blog_corpus is the filtered one
    this function just preprocess the day blog from the saved file(eg:930.txt...) and process and integrate all to one list
    then to compute every word tfidf and extract the feature word using tfidf
    lastly save the top word_tfidf into txt and pickle(20160311/wordDict.txt or 20160311/wordDict.pkl)
    '''
    word_dict = {}
    new_blog_corpus = []
    for doc in blog_corpus:
        tmp = []
        for word in doc:
            if is_word_invalid(word):
                continue
            else:
                tmp.append(word)
                word_dict[word] = 0
        new_blog_corpus.append(tmp)
    print('word_preprocessing-all microblog number and new %d %d' % (len(blog_corpus), len(new_blog_corpus)))
    print('word_preprocessing-all word number %d' % len(word_dict))

    word_tfidf_dict = {}
    for doc in new_blog_corpus:
        score_dict = {}
        for term in doc:
            tfidf = tf_idf(term, doc, new_blog_corpus)
            score_dict[term] = tfidf
        score_list = sorted(score_dict.iteritems(), key=lambda d:d[1], reverse = True)
        if (len(score_list) >= 3):
            for cur in range(3):
                word_tfidf_dict[score_list[cur][0]] = score_list[cur][1]
    word_tfidf_list = []
    # word_tfidf_list = sorted(word_tfidf_dict.iteritems(), key=lambda d:d[1], reverse = False)
    for word in word_tfidf_dict:
        tp = []
        tp.append(word)
        tp.append(word_tfidf_dict[word])
        word_tfidf_list.append(tp)
    print('word_preprocessing-all new word number %d' % len(word_tfidf_list))
    iohelper.save_list2pickle(word_tfidf_list, subdir, 'wordTFDict')
    iohelper.save_list2txt(word_tfidf_list, subdir, 'wordTFDict')
    print('word_preprocessing-save word_list_tfidf success!')

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
def __tf(term, doc, normalize=True):
	if normalize:
		return doc.count(term) / float(len(doc))
	else:
		return doc.count(term) / 1.0

def __idf(term, corpus):
	num_texts_with_term = len([True for text in corpus if term in text])
	try:
		return 1.0 + log(float(len(corpus)) / num_texts_with_term)
	except ZeroDivisionError:
		return 1.0

def tf_idf(term, doc, corpus):
    tf_value = __tf(term, doc)
    idf_value = __idf(term, corpus)
    if tf_value == 0:
        return 0
    else:
        return tf_value * idf_value

# -----------------------------------------------------------------------------
def word_tokenization(tick_blog_list):
    '''
    word tokenization by jieba to list
    return list : [[,], [,], ...]
    '''
    count = 0
    seg_list = []
    try:
        for blog in tick_blog_list:
            if blog != '':
                count += 1
                segments = jieba.cut(blog)
                tmp = []
                for seg in segments:
                    tmp.append(seg)
                seg_list.append(tmp)
    except IOError as e:
        logging.error('IOError %s' % e)
    finally:
        return seg_list

# Python双击可以直接测试
if __name__ == '__main__':
    main()
