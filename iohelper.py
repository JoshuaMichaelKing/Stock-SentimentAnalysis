#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys, codecs, logging
import cPickle as pickle
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

# txt <--> list
def read_txt2list(fname, subdir=None):
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

def save_list2file(sentence_list, filefullpath):
    '''
    save list to txt, add \n to every list member's end
    '''
    f = codecs.open(filefullpath, 'w', 'utf-8')
    for tp in sentence_list:
        f.write(tp + '\n')
    f.close()

def save_list2txt(word_list_tfidf, subdir, fname):
    '''
    save list(element:key-value) to txt, add \n to every list member's end
    '''
    filepath = './Data/' + subdir + '/' + fname + '.txt'
    f = codecs.open(filepath, 'a', 'utf-8')
    for tp in word_list_tfidf:
        f.write(tp[0] + ':' + str(tp[1]) + '\n')
    f.close()

# pickle <--> list
def read_pickle2list(fpath, fname=None):
    '''
    read pickle to list
    '''
    output = ''
    if fname is None:
        output = open(fpath, 'rb')
    else:
        output = open('./Data/' + fpath + '/' + fname + '.pkl', 'rb')
    # Pickle dictionary using protocol 0.
    list_seq = pickle.load(output)
    output.close()
    return list_seq

def save_list2pickle(word_list, fpath, fname=None):
    '''
    save word list to file by pickle
    '''
    output = ''
    if fname is None:
        output = open(fpath, 'wb')
    else:
        output = open('./Data/' + fpath + '/' + fname + '.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(word_list, output)
    output.close()

if __name__ == '__main__':
    main()
