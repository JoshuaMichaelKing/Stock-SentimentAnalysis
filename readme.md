## Stock Index Prediction Based on Sentiment Analysis(基于情感分析的股指预测)

### Data Preprocessing(数据集预处理)
- stocktime.py is to provide the basic open time and close time of A stock(Shanghai Composite Index) in China
- snspy.py(supported by liaoxuefeng) is to get microblog from sina
- weibo.py is to process every blog clean and save it to txt every 5 minutes
- stockindex.py is to get newest market data from sina js server and save list to pickle(shindex_seq.pkl), list contains every 5-min close index

### Reviews Preprocessing(评论的手动标注)
- reviews_preprocessing.py is to seperate all the reviews into two part : positive and negative part(with some handful work)
- save the negative set to ./Reviews/neg_reviews.pkl and the positive set to ./Reviews/pos_reviews.pkl

### Classifier Score(分类器的比较)
- tools : using the package nltk and scikit-learn to implement machine learning method
- using the hand-tag positive and negative reviews to implement machine learning classifier(Logistic Regression, Naive Bayes and Support Vector Machine)

### Sentiment Index Computation(情感系数的计算)
- sentiment.py is to compute index from Comments/20160000/****.txt and save list to pickle(saindex_seq.pkl), list contains every 5 minutes sentiment index

### Correlation Analysis(相关性分析)
- correlation.py is to correlate the sequence of sentiment index and shanghai composite index at a specified day

### Linear Regression(多元线性回归)
- use historical 5-min close index to predict next close index(multi-linear regression)
- plus 5-min sentiment-index into multi-factor and predict again

### Other Infos(配置及文件夹信息)
- config.ini is to write information(App Key, App Secret and Callback URL) to connect sina open api
- Directory Comments is to save all the online comments from sina weibo
- Directory Lexicons is to save lexicon dict
- Directory Reviews is to save tagged positive and negtive comments.
- Directory Sentiment Index is to save everyday 5-min sentiment index time Series
- Directory Stock Index is to save everyday 5-min stock close index time series

### The Classifying Result(分类结果)
#### Not Using Bigrams
- The Lexicons's average precision recall accuracy score is repectively : 0.95 0.93 0.94
- The LR's average precision recall accuracy score is repectively : 0.91 0.90 0.90
- The BernoulliNB's average precision recall accuracy score is repectively : 0.87 0.82 0.85
- The MultinomialNB's average precision recall accuracy score is repectively : 0.87 0.80 0.84
- The LinearSVC's average precision recall accuracy score is repectively : 0.90 0.89 0.90
- The NuSVC's average precision recall accuracy score is repectively : 0.89 0.88 0.89
- The SVC's average precision recall accuracy score is repectively : 0.85 0.85 0.85

#### Using Bigrams
- The Lexicon's average precision recall accuracy score is repectively : 0.95 0.93 0.94
- The LR's average precision recall accuracy score is repectively : 0.91 0.89 0.90
- The BernoulliNB's average precision recall accuracy score is repectively : 0.87 0.81 0.85
- The MultinomialNB's average precision recall accuracy score is repectively : 0.87 0.80 0.84
- The LinearSVC's average precision recall accuracy score is repectively : 0.89 0.88 0.89
- The NuSVC's average precision recall accuracy score is repectively : 0.90 0.88 0.89
- The SVC's average precision recall accuracy score is repectively : 0.85 0.85 0.85

### The Prediction Result(预测结果)
- contains historical data and plus sentiment index(based on lexicons and best-classifier:LR)
- from left to right, the metrics name respectively is : MA, RMSE, MAPE
#### Historical Regression
Test-Metrics :   0.970355  5.105886  0.001043

#### Lexicons-POS
- AVG_UP_DOWN_NUM:50.21%	COEF_AVG:0.55	COEF_MAX:0.89
- use emotion with forward_unit 0
Test-Metrics :   0.963524  5.663620  0.001307
- use emotion with forward_unit 1
Test-Metrics :   0.969921  5.143081  0.001062
- use emotion with forward_unit 2
Test-Metrics :   0.964569  5.581913  0.001198

#### Lexicons-NEG
- AVG_UP_DOWN_NUM:50.21%	COEF_AVG:0.55	COEF_MAX:0.91
- use emotion with forward_unit 0
Test-Metrics :   0.968722  5.244613  0.001122
- use emotion with forward_unit 1
Test-Metrics :   0.969858  5.148468  0.001050
- use emotion with forward_unit 2
Test-Metrics :   0.967904  5.312698  0.001109

#### Lexicons-TOTAL
- AVG_UP_DOWN_NUM:52.29%	COEF_AVG:0.48	COEF_MAX:0.85
- use emotion with forward_unit 0
Test-Metrics :   0.966926  5.393047  0.001120
- use emotion with forward_unit 1(*****)
Test-Metrics :   0.970562  5.088002  0.001040
- use emotion with forward_unit 2
Test-Metrics :   0.968081  5.298080  0.001099

#### Lexicons-LOG
- AVG_UP_DOWN_NUM:52.29%	COEF_AVG:0.48	COEF_MAX:0.87
- use emotion with forward_unit 0
Test-Metrics :   0.969931  5.142224  0.001051
- use emotion with forward_unit 1
Test-Metrics :   0.970319  5.108980  0.001042
- use emotion with forward_unit 2
Test-Metrics :   0.970151  5.123398  0.001057

#### LR-POS
- AVG_UP_DOWN_NUM:50.62%	COEF_AVG:0.54	COEF_MAX:0.90
- use emotion with forward_unit 0(****)
Test-Metrics :   0.970476  5.095428  0.001042
- use emotion with forward_unit 1
Test-Metrics :   0.970361  5.105314  0.001040
- use emotion with forward_unit 2
Test-Metrics :   0.970142  5.124187  0.001053

#### LR-NEG
- AVG_UP_DOWN_NUM:50.62%	COEF_AVG:0.55	COEF_MAX:0.89
- use emotion with forward_unit 0
Test-Metrics :   0.970899  5.058781  0.001039
- use emotion with forward_unit 1
Test-Metrics :   0.970319  5.108953  0.001040
- use emotion with forward_unit 2
Test-Metrics :   0.970280  5.112336  0.001052

#### LR-TOTAL
- AVG_UP_DOWN_NUM:54.38%	COEF_AVG:0.46	COEF_MAX:0.89
- use emotion with forward_unit 0
Test-Metrics :   0.969599  5.170520  0.001049
- use emotion with forward_unit 1
Test-Metrics :   0.970363  5.105129  0.001041
- use emotion with forward_unit 2
Test-Metrics :   0.970068  5.130467  0.001055

#### LR-LOG
- AVG_UP_DOWN_NUM:54.38%	COEF_AVG:0.48	COEF_MAX:0.88
- use emotion with forward_unit 0
Test-Metrics :   0.969922  5.143031  0.001051
- use emotion with forward_unit 1
Test-Metrics :   0.970315  5.109257  0.001042
- use emotion with forward_unit 2
Test-Metrics :   0.970151  5.123417  0.001057

### OpenSource Protocol(开源协议)
- These source code are distributed under the MIT.
- Written By Joshua Guo(1992gq@gmail.com)
