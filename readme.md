## Stock Index Prediction Based on Sentiment Analysis(基于情感分析的股指预测)

### data explore and preprocessing(数据集预处理)
- stocktime.py is to provide the basic open time and close time of A stock(Shanghai Composite Index) in China
- snspy.py(supported by liaoxuefeng) is to get microblog from sina
- weibo.py is to process every blog clean and save it to txt every 5 minutes
- stockindex.py is to get newest market data from sina js server and save list to pickle(shindex_seq.pkl), list contains every 5-min close index

### reviews preprocessing(评论的手动标注)
- reviews_preprocessing.py is to seperate all the reviews into two part : positive and negative part(with some handful work)
- save the negative set to ./Reviews/neg_reviews.pkl and the positive set to ./Reviews/pos_reviews.pkl

### classifier score(分类器的比较)
- tools : using the package nltk and scikit-learn to implement machine learning method
- using the hand-tag positive and negative reviews to implement machine learning classifier(Logistic Regression, Naive Bayes and Support Vector Machine)

### sentiment index computing(情感系数的计算)
- sentiment.py is to compute index from Comments/20160000/****.txt and save list to pickle(saindex_seq.pkl), list contains every 5 minutes sentiment index

### correlation analysis(相关性分析)
- correlation.py is to correlate the sequence of sentiment index and shanghai composite index at a specified day

### linear regression(多元线性回归)
- use historical 5-min close index to predict next close index(multi-linear regression)
- plus 5-min sentiment-index into multi-factor and predict again

### other info(配置及文件夹信息)
- config.ini is to write information(App Key, App Secret and Callback URL) to connect sina open api
- Directory Comments is to save all the online comments from sina weibo
- Directory Lexicons is to save lexicon dict
- Directory Reviews is to save tagged positive and negtive comments.
- Directory Sentiment Index is to save everyday 5-min sentiment index time Series
- Directory Stock Index is to save everyday 5-min stock close index time series

### the classifying result(分类结果)
- not use bigrams
1. The Lexicon's average precision recall accuracy score is repectively : 0.95 0.93 0.94
2. The LR's average precision recall accuracy score is repectively : 0.91 0.89 0.90
3. The GaussianNB's average precision recall accuracy score is repectively : 0.87 0.82 0.85
4. The BernoulliNB's average precision recall accuracy score is repectively : 0.87 0.82 0.85
5. The MultinomiaNB's average precision recall accuracy score is repectively : 0.87 0.82 0.85
6. The LinearSVC's average precision recall accuracy score is repectively : 0.87 0.82 0.85
7. The NuSVC's average precision recall accuracy score is repectively : 0.90 0.89 0.89
8. The SVC's average precision recall accuracy score is repectively : 0.85 0.86 0.85

- use bigrams
1. The Lexicon's average precision recall accuracy score is repectively : 0.95 0.93 0.94
2. The LR's average precision recall accuracy score is repectively : 0.91 0.89 0.90
3. The GaussianNB's average precision recall accuracy score is repectively : 0.87 0.81 0.85
4. The BernoulliNB's average precision recall accuracy score is repectively : 0.87 0.81 0.85
5. The MultinomiaNB's average precision recall accuracy score is repectively : 0.87 0.81 0.85
6. The LinearSVC's average precision recall accuracy score is repectively : 0.87 0.81 0.85
7. The NuSVC's average precision recall accuracy score is repectively : 0.90 0.88 0.89
8. The SVC's average precision recall accuracy score is repectively : 0.85 0.85 0.85

### the prediction result(预测结果)
- not use emotion
1. error_fit_score: 0.996965582925    error_predict_score: 0.970354598799
2. rmse_fit_score: 5.16586476254      rmse_predict_score: 5.10588622678
3. mape_fit_score: 0.00112419981721   mape_predict_score: 0.00104280710265

- use emotion with forward_unit 0
1. error_fit_score: 0.996995821276    error_predict_score: 0.970084925573
2. rmse_fit_score: 5.14006106853      rmse_predict_score: 5.1290568296
3. mape_fit_score: 0.00112849996807   mape_predict_score: 0.00106378219496

- use emotion with forward_unit 1
1. error_fit_score: 0.996972124703    error_predict_score: 0.970275665737
2. rmse_fit_score: 5.1602933173       rmse_predict_score: 5.11267910699
3. mape_fit_score: 0.00112252633156   mape_predict_score: 0.0010481331951

- use emotion with forward_unit 2
1. error_fit_score: 0.996970108219    error_predict_score: 0.97029889752
2. rmse_fit_score: 5.16201134014      rmse_predict_score: 5.11068074653
3. mape_fit_score: 0.00112411317457   mape_predict_score: 0.00104823883078

### OpenSource Protocol(开源协议)
- These source code are distributed under the MIT.
- Written By Joshua Guo(1992gq@gmail.com)
