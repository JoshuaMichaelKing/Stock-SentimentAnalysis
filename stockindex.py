a#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import stocktime as st
import datetime as dt
from datetime import datetime
import time, requests, os, pickle, logging

__version__ = '0.0.1'
__license__ = 'MIT'
__author__ = 'Joshua Guo (1992gq@gmail.com)'

'''
Python get sh000001 day stock data for data analysis.
'''

def main():
    FILE = os.curdir
    logging.basicConfig(filename=os.path.join(FILE,'log.txt'), level=logging.INFO)
    get_stock_index_5_min()

def get_stock_index_5_min():
    '''
    get stock index last price: 9:35~11:30(24) + 13:05~15:00(24) = 48 tick
    '''
    stockid = 's_sh000001'     # 上证综指
    # initialze time data
    todaydate = st.todaydate
    opentime1 = st.opentime1   # 务必在9:35之前启动
    midclose = st.midclose
    opentime2 = st.opentime2
    closetime = st.closetime

    tick_shift = dt.timedelta(seconds=15)
    tick_delta = dt.timedelta(minutes=5)
    tick_now = opentime1
    if datetime.now() > midclose:
        tick_now = opentime2
    index_seq = []
    is_middle_saved = False
    while True:
        try:
            now = datetime.now()
            mid_shift = midclose + tick_shift
            close_shift = closetime + tick_shift
            if (now >= opentime1 and now <= mid_shift) or (now >= opentime2 and now <= close_shift):
                if now.hour == tick_now.hour and now.minute == tick_now.minute:
                    if abs(now.second - tick_now.second) <= 3:
                        sindex = getStockIndexFromSina(stockid)
                        marketdatas = sindex.split(',')
                        index_seq.append(marketdatas[1])
                        tick_now = tick_now + tick_delta
                        save_list2file_pickle(index_seq, todaydate)
                        print('>>>>>>>>>>Save sequence to file by pickle : %s' % index_seq)
            else:
                if now > midclose and now < opentime2:
                    print('>>>>>>>>>>>>>>>Now it is in middle time!')
                    time.sleep(1)
                    if is_middle_saved == False:
                        print('>>>>>>>>>>Save sequence to file by pickle : %s' % index_seq)
                        save_list2file_pickle(index_seq, todaydate)
                        tick_now = opentime2
                        is_middle_saved = True
                    continue
                elif tick_now > closetime:
                    print('>>>>>>>>save stock index to file by pickle : %s' % index_seq)
                    save_list2file_pickle(index_seq, todaydate)
                    break
        except Exception, e:
            print(e)
        finally:
            print('>>>>>>>>>>>>refresh %s %d %d' % (tick_now, now.minute, now.second))
            time.sleep(1)
    print('>>>>>>>>>>>>>stock index sequence collector ending...')

def make_dir_if_not_exist(filepath):
    if os.path.exists(str(filepath)):
        pass
    else:
        os.mkdir(str(filepath))

def save_list2file_pickle(index_seq, subdir):
    '''
    save stock sequence to file by pickle
    '''
    filepath = './Stock Index/' + subdir
    make_dir_if_not_exist(filepath)
    filepath = filepath + '/'
    output = open(filepath + 'shindex_seq.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(index_seq, output)
    output.close()
    logging.info('STOCKINDEX save stock sequence to file by pickle at %s' % datetime.now())

def getStockIndexFromSina(sid):
    url = "http://hq.sinajs.cn/list=" + sid
    result = requests.get(url).content
    source = result.split("\"")
    if source.count >= 1:
        data = source[1].split(",")
        # for dt in data:
        #     print dt
        return source[1]
    return ""

if __name__ == '__main__':
    main()
