#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import os, sys, codecs, logging
from math import log
import datetime as dt

import iohelper
import stocktime as st

reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    neg_tk_lst = iohelper.read_pickle2list('./Reviews/neg_reviews.pkl')
    pos_tk_lst = iohelper.read_pickle2list('./Reviews/pos_reviews.pkl')
    print(neg_tk_lst[0][0])
    print(pos_tk_lst[0][0])

if __name__ == '__main__':
    main()
