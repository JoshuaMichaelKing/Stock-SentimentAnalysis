#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
import os, sys, codecs, logging
try:
    import cPickle as pickle
except:
    import pickle
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
    filepath = './Lexicons/' + fname
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
        filepath = './Comments/' + fname + '.txt'
    else:
        filepath = './Comments/' + subdir + '/' + fname + '.txt'
    readfile = codecs.open(filepath, 'r', 'utf-8')
    output = readfile.readlines()  # 对于小文件可以一下全部读出
    for weibo in output:
        weibo = weibo.replace('\n', '')
        bloglist.append(weibo)
    readfile.close()
    return bloglist

# reviews <---> list
def read_file2list(type):
    '''
    type : pos or neg
    read reviews to list
    '''
    rev_list = []
    filepath = ''
    if type is 'pos':
        filepath = './Reviews/pos_reviews'
    else:
        filepath = './Reviews/neg_reviews'
    readfile = codecs.open(filepath, 'r', 'utf-8')
    output = readfile.readlines()  # 对于小文件可以一下全部读出
    for cmt in output:
        cmt = cmt.replace('\n', '')
        rev_list.append(cmt)
    readfile.close()
    return rev_list

def save_list2file(sentence_list, filefullpath):
    '''
    save list to txt, add \n to every list member's end
    '''
    f = codecs.open(filefullpath, 'w', 'utf-8')
    for tp in sentence_list:
        f.write(tp + '\n')
    f.close()

# pickle <--> list
def read_pickle2objects(fpath, fname=None):
    '''
    read pickle to objects(object, list, set, dict and so on)
    '''
    output = ''
    if fname is None:
        output = open(fpath, 'rb')
    else:
        if fname.startswith('sh'):
            output = open('./Stock Index/' + fpath + '/' + fname + '.pkl', 'rb')
        elif fname.startswith('sa'):
            output = open('./Sentiment Index/' + fpath + '/' + fname + '.pkl', 'rb')
    # Pickle dictionary using protocol 0.
    objects = pickle.load(output)
    output.close()
    return objects

def save_objects2pickle(objects, fpath, fname=None):
    '''
    save objects(object, list, set, dict and so on) to file by pickle
    '''
    output = ''
    if fname is None:
        output = open(fpath, 'wb')
    else:
        output = open('./Comments/' + fpath + '/' + fname + '.pkl', 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(objects, output)
    output.close()

if __name__ == '__main__':
    main()
