## Correlating Stock Index Movement with Investor Sentiment on Social Network

### data explore and preprocessing
- stocktime.py is to provide the basic open time and close time of A stock(Shanghai Composite Index) in China
- snspy.py(supported by liaoxuefeng) is to get microblog
- weibo.py is to process every blog clean and save it to txt every 5 minutes
- stockindex.py is to get newest market data from sina js server and save list to pickle(shindex_seq.pkl), list contains every 5 minutes stock index

### reviews preprocessing
- reviews_preprocessing.py is to seperate all the reviews into two part : positive and negative part(with some handful work)

### sentiment index computing
- sentiment.py is to compute index from Data/20160000/****.txt and save list to pickle(saindex_seq.pkl), list contains every 5 minutes sentiment index
- using the seperated reviews to implement machine learning classifier(Logistic Regression, Naive Bayes and Support Vector Machine)
- using the package nltk and scikit-learn to implement machine learning method

### correlation analysis
- correlation.py is to correlate the sequence of sentiment index and shanghai composite index at a specified day

### other info
- config.ini is to write information(App Key, App Secret and Callback URL) to connect sina open api
- Directory Dictionary is to save lexicon dict
- Directory Reviews is to save tagged positive and negtive comments.

These source code are distributed under the MIT.
Written By Joshua Guo(1992gq@gmail.com)
