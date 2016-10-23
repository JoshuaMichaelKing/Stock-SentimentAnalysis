#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from snspy import APIClient, SinaWeiboMixin      # using SinaWeibo
from datetime import datetime
import datetime as dt
import stocktime as st
import sys, os, time, json, logging, codecs
reload(sys)
sys.setdefaultencoding('utf-8')

__version__ = '0.0.1'
__license__ = 'MIT'
__author__ = 'Joshua Guo (1992gq@gmail.com)'

'''
Python get weibo for text mining. Require Python 2.6/2.7.
'''

APP_KEY = ''            # app key
APP_SECRET = ''      # app secret
CALLBACK_URL = ''  # callback url

def main():

    FILE = os.curdir
    logging.basicConfig(filename=os.path.join(FILE,'log.txt'), level=logging.INFO)

    config_init()
    print('APP_KEY:%s  APP_SECRET:%s  CALLBACK_URL:%s' % (APP_KEY, APP_SECRET, CALLBACK_URL))
    client = APIClient(SinaWeiboMixin, app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=CALLBACK_URL)
    url = client.get_authorize_url()    # redirect the user to 'url'
    print(url)
    # needGetCode = raw_input("Do you need to get code from callback in browser? Please input yes or no!")
    needGetCode = 'no'
    code = ''
    r = {}
    if needGetCode == 'yes':
        code = raw_input("Please input the returned code : ")   # redirect to url and get code
        r = client.request_access_token(code)
        print(r.access_token)  # access token，e.g., abc123xyz456
        print(r.expires)      # token expires time, UNIX timestamp, e.g., 1384826449.252 (10:01 am, 19 Nov 2013, UTC+8:00)
    # 测试版本access_token的时间期限为1天，过期后再重新获取
    access_token = '2.00TD47_Gv7rSwB6b1fbd6797YonR5C'
    expires = 1622967755.0
    client.set_access_token(access_token, expires)

    # After Initialize API Authorization...
    get_blog_and_process(client)
    print('get_blog_and_process is over...')

    # testing code
    # test_get_blog_and_save(client)
    # blogs = u'$上证指数 sh000001$ #聚焦2016年全国两会# 车易拍赶紧的'
    # bloglists = sentence_cut_list(list(blogs))
    # print(''.join(bloglists))

def config_init():
    filename = 'config.ini'
    f = open(filename, 'rb')  #以只读的方式打开
    contents = f.read()
    f.close()

    allconfigs = contents.split('\n')
    config = []
    for cfg in allconfigs:
        if cfg == '':
            continue
        tmp = cfg.replace('\r', '')
        if '#' in tmp:
            continue
        config.append(tmp)
    global APP_KEY
    APP_KEY = config[0]
    global APP_SECRET
    APP_SECRET = config[1]
    global CALLBACK_URL
    CALLBACK_URL = config[2]

def make_dir(filepath):
    if os.path.exists(str(filepath)):
        pass
    else:
        os.mkdir(str(filepath))

def save2Txt(weibodict, filename, dir):
    filepath = './Data/' + dir
    make_dir(filepath)
    filepath = filepath + '/'
    f = codecs.open(filepath + filename + '.txt', 'a', 'utf-8')
    for i in weibodict:
        f.write(weibodict[i] + '\n')
    f.close()
    logging.info('Weibo save2txt success at %s   %s' % (datetime.now(), filename))

def get_all_userid_sina(client):
    '''
    get all user ids from sina and backup to pickle
    '''
    userid = get_users(client, 'gqmk21st')
    userdict = {}
    userdict = print_users_list(userid)
    userdict = read_file_dict()
    print(len(userdict))
    for k in userdict:
        print('dict[%s] =' % k, userdict[k])
    save_dict_file(userdict)
    return userdict

def get_all_userid_pkl():
    '''
    get all user ids from pickle file
    '''
    userdict = {}
    userdict = read_file_dict()
    print(len(userdict))
    for k in userdict:
        print('dict[%s] =' % k, userdict[k])
    return userdict

def print_users_list(ul):
    '''
    print all users info
    '''
    index = 0
    userdict = {}
    for user in ul:
        uid = user["id"]
        ugen = user["gender"]
        uname = user["screen_name"]
        # uloc = user["location"]
        # udesc = user["description"]
        # print('%-6d%-12d%-3s%s' % (index, uid, ugen, uname))
        index += 1
        userdict[uid] = uname
    return userdict

def get_users(client, uname):
    '''
    API : get all users info from sina api
    '''
    fl = []
    next_cursor = 0
    while True:
        raw_fl = client.friendships.friends.get(screen_name=uname, cursor=next_cursor, count=200)
        fl.extend(raw_fl["users"])
        next_cursor += 1
        if next_cursor == 10:
            break
        time.sleep(1)
    return fl

def get_newest_personalweibo(client, id):
    '''
    API : this api is just limited for authorized user
    get designated following friends newest weibo by uid
    '''
    i = 1
    while i < 5:
        i += 1
        jsdict = client.statuses.user_timeline.get(uid=id, page=i)
        for status in jsdict.statuses:
            # print(jsdict.statuses[m].user.id, jsdict.statuses[m])
            print(status.created_at, status.user.id, status.user.name.encode("GBK", 'replace'), status.text.encode("GBK", 'replace'))
        time.sleep(1)

def get_newest_publicweibo(client):
    '''
    get newest friend microblog
    '''
    jsdict = {}
    try:
        jsdict = client.statuses.friends_timeline.get(page=1, count=180)
    except Exception as e:
        logging.error('>>>>>>>>>>get_newest_publicweibo ERROR : %s %s' % (datetime.now(), e))
        print('>>>>GET ERROR')
        del client
        time.sleep(10)
        config_init()
        client = APIClient(SinaWeiboMixin, app_key=APP_KEY, app_secret=APP_SECRET, redirect_uri=CALLBACK_URL)
        url = client.get_authorize_url()    # redirect the user to 'url'
        access_token = '2.00TD47_Gv7rSwBd7c70e9da30PlWbc'
        expires = 1614584749.0
        client.set_access_token(access_token, expires)
        jsdict = get_newest_publicweibo(client)
        logging.error('>>>>>>>>>> %s' % jsdict)
    finally:
        return jsdict


def send_one_message(client):
    '''
    API : send message or picture to sina weibo
    '''
    content = 'Hello World! By Python SDK'
    client.statuses.update.post(status=content)
    # print(client.statuses.upload.post(status=u'test weibo with picture', pic=open('/Users/michael/test.png')))

def sentence_cut_list(sentencelists):
    '''
    data clean process
    to filter some irrelevant signal
    to filter the sentence containing web link
    to filter the words begin with # or $ and ends with # or $
    '''
    cutlist = "[。，！……!《》<>\"':：？?、/\|“”‘’；]★@{}（）{}【】()｛｝（）：,.;、~——+％%`:“”'‘\n\r".decode('utf-8')
    l = []
    length = len(sentencelists)
    i = 0
    is_to_skip = False
    skip_to_num = 0
    for wd in sentencelists:
        if is_to_skip:
            if i == skip_to_num:
                is_to_skip = False
            else:
                i += 1
                continue
        if wd not in cutlist:       # to filter some irrelevant signal
            if wd == '#':     # filter # #
                cursor = i + 1
                while cursor < length:
                    if sentencelists[cursor] == '#':
                        is_to_skip = True
                        skip_to_num = cursor + 1
                        break
                    else:
                        cursor += 1
                        if cursor - i > 100:
                            break
                i += 1
                continue
            elif wd == '$':
                cursor = i + 1
                while cursor < length:
                    if sentencelists[cursor] == '$':
                        is_to_skip = True
                        skip_to_num = cursor + 1
                        break
                    else:
                        cursor += 1
                        if cursor - i > 100:
                            break
                i += 1
                continue
            elif wd == 'h':      # filter the text containing web link http://...
                if (i + 3) < length:
                    if sentencelists[i+1] == 't' and sentencelists[i+2] == 't' and sentencelists[i+3] == 'p':
                        break
            l.append(wd)
        i += 1
    return l

def get_blog_and_process(client):
    # initialze time data
    todaydate = st.todaydate
    opentime1 = st.opentime1    # 建议在9:35之间启动
    midclose = st.midclose
    opentime2 = st.opentime2
    closetime = st.closetime

    tick_delta = dt.timedelta(minutes=5)  # time minute delta, now condsider 5 minutes as a cycle
    tick_start = opentime1 - tick_delta
    tick_end = tick_start
    nowtime =  datetime.now()
    if nowtime < opentime1:
        print('it is before trading day!')
        tick_end = opentime1
        tick_start = tick_end - tick_delta * 3
    elif nowtime > opentime1 and nowtime <= midclose:
        print('it is in the first trading day!')
        minute = nowtime.minute - nowtime.minute % 5
        time_str = st.todaydate + ' ' + str(nowtime.hour) + ':' + str(minute) + ':00'
        tick_end = datetime.strptime(time_str, '%Y%m%d %H:%M:%S')
        tick_start = tick_end - tick_delta
    elif nowtime > midclose and nowtime < opentime2:
        print('it is in the mid time, not trading!')
        tick_end = opentime2
        tick_start = tick_end - tick_delta * 3
    elif nowtime > opentime2 and nowtime <= closetime:
        print('it is in the second trading day!')
        minute = nowtime.minute - nowtime.minute % 5
        time_str = st.todaydate + ' ' + str(nowtime.hour) + ':' + str(minute) + ':00'
        tick_end = datetime.strptime(time_str, '%Y%m%d %H:%M:%S')
        tick_start = tick_end - tick_delta
    else:
        print('it is time after trading time!')
        return
    print('>>>>>>>>>>>>>>>Weibo collector begin! %s %s', (tick_start, tick_end))

    counter = 0
    counter_sina = 1
    cache_weio_dict = {}
    is_set_again = False
    while True:
        now = datetime.now()
        if now > midclose and now < opentime2:
            mid_delta = dt.timedelta(minutes=6)   # abvoid losing the last 5 minutes data persistance
            mid_later = midclose + mid_delta
            if now > mid_later:
                print('>>>>>>>>>>>>>>>Weibo collector middle end! %d', now.second)
                time.sleep(1)
                if is_set_again == False:
                    tick_start = opentime2 - tick_delta * 3
                    tick_end = opentime2
                    is_set_again = True
                continue
        elif now > closetime:
            close_delta = dt.timedelta(minutes=6)   # abvoid losing the last 5 minutes data
            close_later = closetime + close_delta
            if now > close_later:
                print('>>>>>>>>>>>>>>>Weibo collector end!')
                break

        # 后移一个tick持久化数据，考虑到新浪微博返回信息的滞后性
        tmp = tick_end + tick_delta
        if now >= tmp:
            hour = tick_end.hour
            minute = tick_end.minute
            # flush the cache weibo text to txt file in disk
            filename = str(hour * 100 + minute)
            save2Txt(cache_weio_dict, filename, todaydate)
            cache_weio_dict.clear()    # clear the cache for next saving dict
            print('>>>>>>>>>>>tick start and end! %s %s', (tick_start, tick_end))
            tick_start = tick_end
            tick_end = tmp
            print('>>>>>>>>>>>tick start and end! %s %s', (tick_start, tick_end))

        if counter_sina == 1:
            counter += 1
            jsdict = get_newest_publicweibo(client)
            for status in jsdict.statuses:
                datetime_array = status.created_at.split(' ')
                current_str = todaydate + ' ' + datetime_array[3]
                current_time = datetime.strptime(current_str, '%Y%m%d %H:%M:%S')
                if current_time >= tick_start and current_time <= tick_end:
                    cut_list = sentence_cut_list(list(status.text))
                    sentence = ('').join(cut_list)
                    cache_weio_dict[status.id] = sentence
        counter_sina += 1
        if counter_sina >= 90:
            counter_sina = 1
        print('counter : %d %d %d' % (counter_sina, counter, len(cache_weio_dict)))
        time.sleep(1)

def test_get_blog_and_save(client):
    '''
    just testing
    '''
    cache_weio_dict = {}
    tick_start = datetime.strptime(st.todaydate + ' 19:40:00', '%Y%m%d %H:%M:%S')
    tick_end = datetime.strptime(st.todaydate + ' 20:00:00', '%Y%m%d %H:%M:%S')
    jsdict = get_newest_publicweibo(client)
    for status in jsdict.statuses:
        datetime_array = status.created_at.split(' ')
        current_str = st.todaydate + ' ' + datetime_array[3]
        current_time = datetime.strptime(current_str, '%Y%m%d %H:%M:%S')
        if current_time >= tick_start and current_time <= tick_end:
            cut_list = sentence_cut_list(list(status.text))
            sentence = ('').join(cut_list)
            cache_weio_dict[status.id] = sentence
    if len(cache_weio_dict) == 0:
        print('microblog zero')
    else:
        print(len(cache_weio_dict))
    save2Txt(cache_weio_dict, 'text', st.todaydate)

if __name__ == '__main__':
    main()
