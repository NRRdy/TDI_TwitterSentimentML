# -*- coding: utf-8 -*-
"""
In this script we import a public dataset of 1,600,000 tweets that have been labeled as evoking either 'positive' or 'negative' sentiment
The dataset can be found here: https://www.kaggle.com/kazanova/sentiment140
We subset 10,000 observations of the data, preprocess the tweet text, and fit a K Nearest Neighbor machine learning model to the subsetted data
"""
# import libraries/packages/functions/etc
import re
import string
import collections

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
import nltk
nltk.download('stopwords')
english_stopwords = stopwords.words("english")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# import data
TDIdir = "/Users/nrr5025/Desktop/TDI Project/"
EmoData = pd.read_csv(TDIdir + "training.1600000.processed.noemoticon.csv", header = None, names = ["sentiment", "id", "time", "query", "user", "tweet"])
EmoData['sentimentNumber'] = EmoData['sentiment'].replace(4, 1)
EmoData['sentimentName'] = np.where(EmoData['sentimentNumber'] == 1, "positive", "negative")

# need to select only a subset of 10,000 observations for now (taking too long with full dataframe)
#first 5000 rows
EmoData_top = EmoData.head(5000)
#last 5000 rows
EmoData_bottom = EmoData.tail(5000)
#combine data
EmoData = EmoData_top.append(EmoData_bottom)

# preprocess tweets
def PrepTweet(tweet):
    
    #define some variables
    START_OF_LINE = r"^"
    OPTIONAL = "?"
    ANYTHING = "."
    ZERO_OR_MORE = "*"
    ONE_OR_MORE = "+"
    SPACE = "\s"
    SPACES = SPACE + ONE_OR_MORE
    NOT_SPACE = "[^\s]" + ONE_OR_MORE
    EVERYTHING_OR_NOTHING = ANYTHING + ZERO_OR_MORE
    ERASE = ""
    FORWARD_SLASH = "\/"
    NEWLINES = r"[\r\n]"
    
    #remove retweet label
    RE_TWEET = START_OF_LINE + "RT" + SPACES
    tweet = re.sub(RE_TWEET, ERASE, tweet)
    
    #remove hyperlinks
    HYPERLINKS = ("http" + "s" + OPTIONAL + ":" + FORWARD_SLASH + FORWARD_SLASH + NOT_SPACE + NEWLINES + ZERO_OR_MORE)
    tweet = re.sub(HYPERLINKS, ERASE, tweet)
    
    #remove hashtags
    HASH = "#"
    tweet = re.sub(HASH, ERASE, tweet)
    
    #tokenize
    tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)
    tweet = tokenizer.tokenize(tweet)
    
    #remove stop words and punctuation
    tweet = [word for word in tweet if (word not in english_stopwords and word not in string.punctuation)]
    
    #stem
    stemmer = PorterStemmer()
    tweet = [stemmer.stem(word) for word in tweet]
    
    return tweet


EmoData['tweetPrep'] = EmoData['tweet'].apply(PrepTweet)


# create counts of all Ngrams in EmoData
NgramCounter = collections.Counter()
N = 2 #size of Ngram

for tP in EmoData['tweetPrep']:
    ng = ngrams(tP, N)
    NgramCounter.update(ng)
    
# create dataframe with M most common Ngrams as columns
##this will only work for bigrams, will have to think how to make it scalable
M = 1000
M_Ngrams = NgramCounter.most_common(M)
ColNames = []

for NG in M_Ngrams:
    ColNames.append(NG[0][0] + "_" + NG[0][1])
    
TweetFeats = pd.DataFrame(columns = ColNames)

# extract frequency for each M Ngram for each tweet
##again, this will only work for bigrams as written
for idx, tP in EmoData['tweetPrep'].iteritems():
    TweetFeats.loc[len(TweetFeats)] = 0
    ng = collections.Counter(ngrams(tP, N))
    for key1, key2 in ng:
        cn = key1 + "_" + key2
        if cn in TweetFeats.columns:
            TweetFeats.iloc[idx].loc[cn] = ng[(key1, key2)]
            
TweetFeats.to_csv(TDIdir + "SentimentTwitterData10000_features.csv", index = False)

## K Nearest Neighbors Classifier ##

# our features - X
X = TweetFeats

# our label - y
y = EmoData['sentimentNumber']

#instantiate classifier
knn = KNeighborsClassifier()

# split data into training and test for final holdout set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state = 27)

# hyperparameter tuning using GridSearchCV
param_grid = {'n_neighbors' : np.arrange(1, 50)}
knn_cv = GridSearchCV(knn, param_grid, cv = 5)
knn_cv.fit(X_train, y_train)
# Print the optimal parameters and best score
print("Tuned k Parameter: {}".format(knn_cv.best_params_))
print("Tuned Accuracy: {}".format(knn_cv.best_score_))

# predict on the test set
y_pred = knn_cv.predict(X_test)
# accuracy
knn_acc = knn_cv.score(X_test, y_test)
# mean squared error
knn_mse = mean_squared_error(y_test, y_pred)

# compute predicted probabilities
y_pred_prob = knn_cv.predict_proba(X_test)[:,1]

# generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# compute cross-validated AUC scores
cv_auc = cross_val_score(knn_cv, X, y, cv = 5, scoring = "roc_auc")

# print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))

# plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

#save file
filename = TDIdir + 'Emo10000_1000bigrams_knn_k5.sav'
pickle.dump(knn_cv, open(filename, 'wb'))


