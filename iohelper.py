#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys, codecs, logging, pickle
reload(sys)
sys.setdefaultencoding('utf-8')

__version__ = '0.0.1'
__license__ = 'MIT'
__author__ = 'Joshua Guo (1992gq@gmail.com)'

'''
Python IO Operation: to save and read txt or pickle; to judge the directory exists.
'''

def main():
    pass

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

def read_pickle2list(filefullpath):
    '''
    read pickle to word list and get tfidf
    '''
    output = open(filefullpath, 'rb')
    # Pickle dictionary using protocol 0.
    word_seq = pickle.load(output)
    output.close()
    return word_seq

def save_list2file_pickle(word_list_tfidf, filefullpath):
    '''
    save word list and tfidf to file by pickle
    '''
    output = open(filefullpath, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(word_list_tfidf, output)
    output.close()

def save_list2file(word_list_tfidf, filefullpath):
    '''
    save list to txt, add \n to every list member's end
    '''
    f = codecs.open(filefullpath, 'a', 'utf-8')
    for tp in word_list_tfidf:
        f.write(tp[0] + ':' + str(tp[1]) + '\n')
    f.close()

def read_blog2list(fname, subdir=None):
    '''
    read txt to list
    '''
    bloglist = []
    filepath = ''
    if subdir is None:
        filepath = '.\\Data\\' + fname
    else:
        filepath = '.\\Data\\' + subdir + '\\' + fname
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

def word_tokenization(file, subdir=None):
    '''
    word tokenization by jieba to list
    '''
    count = 0
    seg_list = []
    try:
        daybloglist = read_file2list(file, subdir)
        for blog in daybloglist:
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
