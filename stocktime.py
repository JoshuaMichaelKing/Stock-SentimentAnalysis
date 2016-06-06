#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

__version__ = '0.0.1'
__license__ = 'MIT'
__author__ = 'Joshua Guo (1992gq@gmail.com)'

'''
Python A stock time info.
'''

from datetime import date
from datetime import datetime

todaydate = date.today().strftime('%Y%m%d')
_opentime1_str = todaydate + ' 9:35:00'
_midclosetime_str = todaydate + ' 11:30:00'
_opentime2_str = todaydate + ' 13:05:00'
_closetime_str = todaydate + ' 15:00:00'

opentime1 = datetime.strptime(_opentime1_str, '%Y%m%d %H:%M:%S')
midclose = datetime.strptime(_midclosetime_str, '%Y%m%d %H:%M:%S')
opentime2 = datetime.strptime(_opentime2_str, '%Y%m%d %H:%M:%S')
closetime = datetime.strptime(_closetime_str, '%Y%m%d %H:%M:%S')

def main():
    pass

if __name__ == '__main__':
    main()
