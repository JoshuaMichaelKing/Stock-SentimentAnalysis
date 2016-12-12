#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os, sys, codecs, logging
from math import log
import datetime
import itertools
from random import shuffle
from itertools import chain

from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import iohelper
import reviews_preprocessing as rp

reload(sys)
sys.setdefaultencoding('utf-8')

__version__ = '0.0.1'
__license__ = 'MIT'
__author__ = 'Joshua Guo (1992gq@gmail.com)'

'''
Machine Learning Sentiment Classifier : Feature Selection(NLTK 3.x Version)
'''

best_words = set()

def main():
    pos_tk_lst = iohelper.read_pickle2objects('./Reviews/pos_reviews.pkl')
    neg_tk_lst = iohelper.read_pickle2objects('./Reviews/neg_reviews.pkl')
    # 使评论集合随机分布
    shuffle(pos_tk_lst)
    shuffle(neg_tk_lst)
    posWords = list(itertools.chain(*pos_tk_lst)) #把多维数组解链成一维数组
    negWords = list(itertools.chain(*neg_tk_lst)) #同理

    # 二选一(前面是所有词，后面是所有词+双词，基于卡方检验疾进行特征选择)
    # print('1.Word Feature Selection-Chi-sq!')
    # word_scores = create_word_scores(posWords, negWords)
    print('2.Word_Plus_Bigram Feature Selection-Chi-sq!')
    pos_tk_lst = words_plus_bigram(pos_tk_lst)
    neg_tk_lst = words_plus_bigram(neg_tk_lst)
    word_scores = create_word_bigram_scores(posWords, negWords)

    global best_words
    best_words = find_best_words(word_scores, 1500)
    iohelper.save_objects2pickle(best_words, './Reviews/best_words.pkl')

    posFeatures = pos_features(pos_tk_lst, best_word_features)   # [[{'':True, '':True,...}, 'pos'], [{'':True, '':True,...}, 'neg']]
    negFeatures = neg_features(neg_tk_lst, best_word_features)
    print('POS_FEATURES_LENGTH %d\tNEG_FEATURES_LENGTH %d' % (len(posFeatures), len(negFeatures)))
    assert len(posFeatures) == len(negFeatures)
    print ('-------------------------------------------------')

    Classifier_Type = ['Lexicons', 'LR', 'BernoulliNB', 'MultinomialNB', 'LinearSVC', 'NuSVC']      # 'SVC' IS CANCELLED
    (pos_lexicon_dict, neg_lexicon_dict) = rp.load_sentiment_lexicon()

    # 10_fold_cross-validation(10折交叉验证)
    cut_size = int(len(posFeatures) * 0.9)
    offset_size = len(posFeatures) - cut_size
    avg_scores = {}
    avg_precision = {}
    avg_recall = {}
    avg_time = {}
    for tp in Classifier_Type:
        avg_scores[tp] = 0.0
        avg_precision[tp] = 0.0
        avg_recall[tp] = 0.0
        avg_time[tp] = 0.0
    posTmp = []
    negTmp = []
    # 比较不同分类器的效果(主要分为基于情感词典的和基于监督式学习的)
    for tp in Classifier_Type:
        precision = 0.0
        recall = 0.0
        score = 0.0
        time = 0.0
        if tp == 'Lexicons':
            posTmp = posFeatures
            negTmp = negFeatures
            posFeatures = pos_tk_lst
            negFeatures = neg_tk_lst

        print ('Classifier_Type : %s' % (tp))
        for k in range(1, 11):
            test_list = posFeatures[(k-1)*offset_size:k*offset_size] + negFeatures[(k-1)*offset_size:k*offset_size]
            if k == 1:
                train_list = posFeatures[k*offset_size:] + negFeatures[k*offset_size:]
            elif k == 10:
                train_list = posFeatures[:(k-1)*offset_size] +  negFeatures[:(k-1)*offset_size]
            else:
                train_list = posFeatures[:(k-1)*offset_size] + posFeatures[k*offset_size:] + negFeatures[:(k-1)*offset_size] + negFeatures[k*offset_size:]

            if tp == 'Lexicons':
                test = test_list
                test_tag = ['pos' for i in range(offset_size)]
                test_tag.extend(['neg' for i in range(offset_size)])
                time, precision, recall, score = sentiment_lexicon_score(pos_lexicon_dict, neg_lexicon_dict, test, test_tag)
            else:
                test, test_tag = zip(*test_list)  # 将内部的元素list(dict和string)分解成两类tuple({}, {}, {},...)和('pos', 'pos', 'neg', ...)
                if tp == 'LR':
                    time, precision, recall, score = classifier_score(tp, LogisticRegression(), train_list, test, test_tag)
                elif tp == 'BernoulliNB':
                    time, precision, recall, score = classifier_score(tp, BernoulliNB(), train_list, test, test_tag)
                elif tp == 'MultinomialNB':
                    time, precision, recall, score = classifier_score(tp, MultinomialNB(), train_list, test, test_tag)
                elif tp == 'LinearSVC':
                    time, precision, recall, score = classifier_score(tp, LinearSVC(), train_list, test, test_tag)
                elif tp == 'NuSVC':
                    time, precision, recall, score = classifier_score(tp, NuSVC(probability=True), train_list, test, test_tag)
                elif tp == 'SVC':
                    precision, recall, score = classifier_score(tp, SVC(gamma=0.001, C=100., kernel='linear', probability=True), train_list, test, test_tag)
            avg_scores[tp] += score
            avg_precision[tp] += precision
            avg_recall[tp] += recall
            avg_time[tp] += time
            print ('The precision recall accuracy score and training time is repectively : %f %f %f %f' % (precision, recall, score, time))
        if tp == 'Lexicons':
            posFeatures = posTmp
            negFeatures = negTmp
            posTmp = []
            posTmp = []
        print ('-------------------------------------------------')
    for tp in Classifier_Type:
        avg_scores[tp] = avg_scores[tp] / 10
        avg_precision[tp] = avg_precision[tp] / 10
        avg_recall[tp] = avg_recall[tp] / 10
        avg_time[tp] = avg_time[tp] / 10
        print ("The %s\'s average precision recall accuracy score and training time is repectively : %.2f %.2f %.2f %.2f" % \
            (tp, avg_precision[tp], avg_recall[tp], avg_scores[tp], avg_time[tp]))
    print ("The End!")

#------------------------------------------------------------------------------
def sentiment_lexicon_score(pos_lexicon_dict, neg_lexicon_dict, test, test_tag):
    '''
    Sentiment Lexicon Score
    Input Type : [[,], [,], ...]
    Output:pos_precision, pos_recall, accuracy_score
    '''
    if type(test) is not type([]):
        raise TypeError("There is a type error","input test should be list!")

    starttime = datetime.datetime.now()
    pred = []
    for blog_lst in test:
        score = rp.sentiment_logarithm_estimation(pos_lexicon_dict, neg_lexicon_dict, blog_lst)
        if score > 0:
            pred.append('pos')
        else:
            pred.append('neg')

    y_true = [1 if tag == 'pos' else 0 for tag in test_tag]
    y_pred = [1 if tag == 'pos' else 0 for tag in pred]
    pos_precision = precision_score(y_true, y_pred)
    pos_recall = recall_score(y_true, y_pred)
    endtime = datetime.datetime.now()
    interval = (endtime - starttime).microseconds
    interval = interval / 100
    return interval, pos_precision, pos_recall, accuracy_score(test_tag, pred)

#------------------------------------------------------------------------------
def classifier_score(tp, classifier, train_list, test, test_tag):
    '''
    传入分类器进行分类
    Output:pos_precision, pos_recall, accuracy_score
    '''
    starttime = datetime.datetime.now()
    classifier = SklearnClassifier(classifier)
    classifier.train(train_list)
    iohelper.save_objects2pickle(classifier, './Reviews/' + tp + '.pkl')
    pred = classifier.classify_many(test)  # 返回的是结果集的list
    y_true = [1 if tag == 'pos' else 0 for tag in test_tag]
    y_pred = [1 if tag == 'pos' else 0 for tag in pred]
    pos_precision = precision_score(y_true, y_pred)
    pos_recall = recall_score(y_true, y_pred)
    endtime = datetime.datetime.now()
    interval = (endtime - starttime).microseconds
    interval = interval / 100
    return interval, pos_precision, pos_recall, accuracy_score(test_tag, pred)

#------------------------------------------------------------------------------
def words_plus_bigram(tk_lst, score_fn=BigramAssocMeasures.chi_sq):
    '''
    所有词+双词
    每句评论里面抽取双词加到该句列表里
    '''
    for words in tk_lst:
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigrams = bigram_finder.nbest(score_fn, 10)
        words.extend(bigrams)
    return tk_lst

def find_best_words(word_scores, num):
    '''
    把词按信息量倒序排序
    num是特征的维度，是可以不断调整直至最优的
    '''
    best_values = sorted(word_scores.iteritems(), key = lambda (w, s): s, reverse = True)[:num]
    best_words = set([w for w, s in best_values])
    return best_words

def best_word_features(words):
    '''
    根据信息量最丰富的词进行筛选
    '''
    global best_words
    return dict([(word, True) for word in words if word in best_words])

def pos_features(pos, feature_selection_method):
    '''
    积极特征集合，格式为[[,...], [,...], ...]
    传入的是特征选择函数
    输出list,格式为[[{},'pos'], [{},'pos'], ...]
    '''
    posFeatures = []
    for i in pos:
        posWords = [feature_selection_method(i), 'pos']
        posFeatures.append(posWords)
    return posFeatures

def neg_features(neg, feature_selection_method):
    '''
    消极特征集合，格式为[[,...], [,...], ...]
    传入的是特征选择函数
    输出list,格式为[[{},'neg'], [{}, 'neg'], ...]
    '''
    negFeatures = []
    for j in neg:
        negWords = [feature_selection_method(j), 'neg']
        negFeatures.append(negWords)
    return negFeatures

#------------------------------------------------------------------------------
def create_word_scores(posWords, negWords, score_method = BigramAssocMeasures.chi_sq):
    '''
    以单独一个词来统计词的信息量
    '''
    word_fd = FreqDist() #可统计所有词的词频
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N() #积极评论中词的数量
    neg_word_count = cond_word_fd['neg'].N() #消极评论中词的数量
    total_word_count = pos_word_count + neg_word_count
    print("IN_POSWORD_NUMS : %d\tIN_NEGWORD_NUMS : %d" % (pos_word_count, neg_word_count))

    #默认使用卡方统计量，这里也可以计算互信息等其它统计量
    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = score_method(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = score_method(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量
    return word_scores #键值对－词：词的信息量

#------------------------------------------------------------------------------
def create_word_bigram_scores(posWords, negWords, score_method = BigramAssocMeasures.chi_sq):
    '''
    以双词来统计词的信息量
    '''
    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(score_method, 5000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(score_method, 5000)
    pos = posWords + posBigrams #词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count
    print("BIGRAM_IN_POSWORD_NUMS : %d\tBIGRAM_IN_NEGWORD_NUMS : %d" % (pos_word_count, neg_word_count))

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = score_method(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = score_method(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores

if __name__ == '__main__':
    main()
