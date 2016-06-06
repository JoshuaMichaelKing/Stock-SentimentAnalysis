## Correlating Stock Index Movement with Investor Sentiment on Social Network by Python 2.7

### data explore and preprocessing
- stocktime.py to provide the basic open time and close time of A stock in China
- snspy.py(by liaoxuefeng) to get microblog
- weibo.py to process every blog clean and save it to txt every 5 minutes
- stockindex.py to get newest market data from sina js server and save list to pickle(shindex_seq.pkl), list contains every 5 minutes stock index

### sentiment index computing
- sentiment.py to compute index from Data/20160000/****.txt and save list to pickle(saindex_seq.pkl), list contains every 5 minutes sentiment index

### correlation analysis
- correlation.py to correlate the sequence of sentiment index and shanghai composite index at a specified day

### other info
- config.ini to write information(App Key, App Secret and Callback URL) to connect sina open api
- directory Data to save files

These source code are distributed under the MIT.
Written By Joshua Guo(1992gq@gmail.com)
