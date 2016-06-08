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
Python : Feature Extraction and Sentiment Index Computing.
'''

_subdir = '20160329'

def main():
    FILE = os.curdir
    logging.basicConfig(filename=os.path.join(FILE,'log.txt'), level=logging.ERROR)
    print('correlation date %s', _subdir)
    # sentiment_lexicon_compute()

def sentiment_ml_compute():
    pass

def sentiment_lexicon_compute():
    '''
    1. select the words to construct dictionary
    2. compute sentiment index according to the stock-oriented dictionary
    '''
    # loading postive and negtive sentiment lexicon
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

    print('pos_lexicon_dict length : %d' % len(pos_lexicon_dict))
    print('neg_lexicon_dict length : %d' % len(neg_lexicon_dict))

    opentime1 = st.opentime1
    midclose = st.midclose
    opentime2 = st.opentime2
    closetime = st.closetime

    tick_delta = dt.timedelta(minutes=5)
    tick_now = opentime1

    blog_corpus = []
    sentiment_index = []
    status = raw_input("Construct dictionary or compute sentiment index? Please input yes(feature extraction) or no(compute sentiment index)!")
    isPrint = False
    while True:
        if (tick_now >= opentime1 and tick_now <= midclose) or (tick_now >= opentime2 and tick_now <= closetime):
            hour = tick_now.hour
            minute = tick_now.minute
            if hour == 13 and minute == 45:
                isPrint = False  # use the flag to print the word score in sentiment computing
            else:
                isPrint = False
            fname = str(hour * 100 + minute)
            tick_blog_list = word_tokenization(fname, _subdir)
            blog_corpus.extend(tick_blog_list)
            tick_now += tick_delta
            # Compute Sentiment Index
            if status != 'yes':
                tick_value_tmp = sentiment_compute_logarithm(pos_lexicon_dict, neg_lexicon_dict, tick_blog_list, isPrint)
                sentiment_index.append(tick_value_tmp)
        elif tick_now > midclose and tick_now < opentime2:
            tick_now = opentime2
        elif tick_now > closetime:
            break
    # not necessary if you have processed it to word_tfidf list txt pkl
    if status == 'yes':
        word_preprocessing(blog_corpus)
    else:
        iohelper.save_list2pickle(sentiment_index, _subdir, 'saindex_seq')
        print('save_list2pickle success! %d %s' % (len(sentiment_index), _subdir))

def sentiment_compute_average(pos_lexicon_dict, neg_lexicon_dict, tick_blog_segments, isPrint):
    '''
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
                    print("%s +%d" % (word, pos_lexicon_dict[word]))
                sentence_count += pos_lexicon_dict[word]
            elif word in neg_lexicon_dict:
                if isPrint:
                    print("%s -%d" % (word, neg_lexicon_dict[word]))
                sentence_count -= neg_lexicon_dict[word]
        index_list.append(sentence_count)
    if len(index_list) != 0:
        tick_value_tmp = sum(index_list) / float(len(index_list))
    print('%f' % tick_value_tmp)
    return tick_value_tmp

def sentiment_compute_logarithm(pos_lexicon_dict, neg_lexicon_dict, tick_blog_segments, isPrint = False):
    '''
    using ln((1+pos)/(1+neg)) formula
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
                if isPrint:
                    print("%s +%d" % (word, pos_lexicon_dict[word]))
                pos_count += pos_lexicon_dict[word]
            elif word in neg_lexicon_dict:
                if isPrint:
                    print("%s -%d" % (word, neg_lexicon_dict[word]))
                neg_count += neg_lexicon_dict[word]
        pos_list.append(pos_count)
        neg_list.append(neg_count)
    tick_value_tmp = log(float(1 + sum(pos_list)) / float(1 + sum(neg_list)))
    print('%f' % tick_value_tmp)
    return tick_value_tmp

def word_preprocessing(blog_corpus):
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
    print('all microblog number and new %d %d' % (len(blog_corpus), len(new_blog_corpus)))
    print('all word number %d' % len(word_dict))

    new_word_dict = {}
    for doc in new_blog_corpus:
        score_dict = {}
        for term in doc:
            tfidf = tf_idf(term, doc, new_blog_corpus)
            score_dict[term] = tfidf
        score_list = sorted(score_dict.iteritems(), key=lambda d:d[1], reverse = True)
        cnt = 0
        for tp in score_list:
            if cnt < 3:
                if tp[0] in new_word_dict:
                    if new_word_dict[tp[0]] < tp[1]:
                        new_word_dict[tp[0]] = tp[1]
                else:
                    new_word_dict[tp[0]] = tp[1]
                cnt += 1
    new_word_list = sorted(new_word_dict.iteritems(), key=lambda d:d[1], reverse = False)
    print('new all word number %d' % len(new_word_list))
    iohelper.save_list2pickle(new_word_list, _subdir, 'wordDict')
    iohelper.save_list2txt(new_word_list, _subdir, 'wordDict')
    print('save word_list_tfidf success!')

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

def tf(term, doc, normalize=True):
	if normalize:
		return doc.count(term) / float(len(doc))
	else:
		return doc.count(term) / 1.0

def idf(term, corpus):
	num_texts_with_term = len([True for text in corpus if term in text])
	try:
		return 1.0 + log(float(len(corpus)) / num_texts_with_term)
	except ZeroDivisionError:
		return 1.0

def tf_idf(term, doc, corpus):
    tf_value = tf(term, doc)
    idf_value = idf(term, corpus)
    if tf_value == 0:
        return 0
    else:
        return tf_value * idf_value

def bag_of_words(words):
    return dict([(word, True) for word in words])

def word_tokenization(fname, subdir=None):
    '''
    word tokenization by jieba to list
    return list : [[,], [,], ...]
    '''
    count = 0
    seg_list = []
    try:
        tick_blog_list = iohelper.read_txt2list(fname, subdir)
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

if __name__ == '__main__':
    main()
