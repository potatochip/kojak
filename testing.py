import numpy as np
import pandas as pd
import data_grab
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
import json
from textblob import TextBlob
from sklearn.cross_validation import cross_val_score
import metrics
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import text_processors
from multiprocessing import Pool
from progressbar import ProgressBar
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from time import time
from itertools import combinations
import operator

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA



def contest_metric(numpy_array_predictions, numpy_array_actual_values):
    return metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS)

def raw_scoring(X, y, pipeline):
    '''since cross_val_score doesn't allow you to round the results beforehand'''
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42)

    p1 = pipeline.fit(xtrain, ytrain['score_lvl_1']).predict(xtest)
    score1 = accuracy_score(ytest['score_lvl_1'], np.round(p1))
    print("Level 1 accuracy score of {}".format(score1))
    p2 = pipeline.fit(xtrain, ytrain['score_lvl_2']).predict(xtest)
    score2 = accuracy_score(ytest['score_lvl_2'], np.round(p2))
    print("Level 2 accuracy score of {}".format(score2))
    p3 = pipeline.fit(xtrain, ytrain['score_lvl_3']).predict(xtest)
    score3 = accuracy_score(ytest['score_lvl_3'], np.round(p3))
    print("Level 3 accuracy score of {}".format(score3))

    results = np.dstack((p1, p2, p3))[0]
    score = contest_metric(np.round(results), np.array(ytest))
    print("Contest score of {}".format(score))

    rounded = np.clip(np.round(results), 0, np.inf)
    compare = pd.concat([pd.DataFrame(np.concatenate((results, rounded), axis=1)), ytest.reset_index(drop=True)], axis=1)
    compare.columns = ['pred1','pred2','pred3','round1','round2','round3','true1','true2','true3']
    compare['offset1'] = compare.round1-compare.true1
    compare['offset2'] = compare.round2-compare.true2
    compare['offset3'] = compare.round3-compare.true3

    return score1, score2, score3, score, compare.head(10)

def extract_features(df):
    features = df.drop(['score_lvl_1', 'score_lvl_2', 'score_lvl_3'], axis=1)
    response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.float64)  #for numerical progression
    # response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.int8)  # for categorical response
    return features, response


def multi_feature_test(X, y, pipeline, feature_list):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42)

    combo_list = []
    for num in range(1, len(feature_list)+1):
        combo_list.extend([list(i) for i in combinations(feature_list, num)])

    temp_dict = {}
    for features in combo_list:
        p1 = pipeline.fit(xtrain[features], ytrain['score_lvl_1']).predict(xtest[features])
        score = accuracy_score(ytest['score_lvl_1'], np.round(p1))
        temp_dict.update({tuple(features): score})
        print("Level 1 accuracy score of {} for {}".format(score, features))
    return temp_dict


def make_bins(df, bin_size=30):
    # time delta bins
    tdmax = df.review_delta.max()
    tdmin = df.review_delta.min()
    df['review_delta_bin'] = pd.cut(df["review_delta"], np.arange(tdmin, tdmax, bin_size))
    df['review_delta_bin_codes'] = df.review_delta_bin.astype('category').cat.codes
    tdmax = df.previous_inspection_delta.max()
    tdmin = df.previous_inspection_delta.min()
    df['previous_inspection_delta_bin'] = pd.cut(df["previous_inspection_delta"], np.arange(tdmin-1, tdmax, bin_size))
    df['previous_inspection_delta_bin_codes'] = df.previous_inspection_delta_bin.astype('category').cat.codes
    return df


def test1(df):
    '''testing multiple models'''
    X, y = extract_features(df)

    tfidf = joblib.load( 'pickle_jar/tfidf_preprocessed_ngram3_sublinear_1mil')

    # set classifiers to test
    estimator_list = [
        MultinomialNB(),
        SGDClassifier(n_jobs=-1),
        ]

    for estimator in estimator_list:
        print(estimator)
        pipeline = Pipeline([
                # ('decomp', TruncatedSVD(n_components=5, random_state=42)),
                # ('decomp', PCA(n_components=2)),
                # ('scaler', StandardScaler()),
                # ('normalizer', Normalizer(norm='l2')), #  for text classification and clustering
                ('scaler', StandardScaler(with_mean=False)), #  for sparse matrix
                ('clf', estimator),
        ])

        raw_scoring(X, y, pipeline)


def test2(df):
    '''feature selection'''
    model_features = ['review_stars', 'review_delta', 'previous_inspection_delta', 'polarity', 'subjectivity', 'neg', 'pos', 'neu', 'compound']
    X, y = extract_features(df[model_features + scores].dropna())
    estimator = SGDClassifier(n_jobs=-1, random_state=42)
    pipeline = Pipeline([
        ('normalizer', Normalizer()),
#         ('normalizer', Normalizer(norm='l2')), #  for text classification and clustering
        ('scaler', StandardScaler()),
        ('clf', estimator),
        ])

    test_dict = multi_feature_test(X, y, pipeline, model_features)
    ranked = sorted(test_dict.items(), key=operator.itemgetter(1))
    print(ranked)
    print("***")
    print(ranked[:-20:-1])
    # print(ranked[0:20])


def test3(df):
    '''test different bin sizes for time deltas
        with perceptron and no binning. scores of .07700, .69218, .39246, 1.3246
        highest scores with bin of 90 and scores of 0.11099, 0.67774, 0.47258, 1.2672
        with SGD and polarity binning isnt any better
    '''
    model_features = ['review_delta_bin_codes', 'previous_inspection_delta_bin_codes', 'polarity']
    estimator = Perceptron(n_jobs=-1, random_state=42)
    pipeline = Pipeline([
        ('normalizer', Normalizer()),
#         ('normalizer', Normalizer(norm='l2')), #  for text classification and clustering
        ('scaler', StandardScaler()),
        ('clf', estimator),
        ])

    test_dict = {}
    for i in np.round(np.linspace(1, 100, 40)):
        df = make_bins(df, i)
        X, y = extract_features(df[model_features + scores].dropna())
        print("bin size: {}".format(i))
        s1, s2, s3, s4, c = raw_scoring(X, y, pipeline)
        test_dict.update({i: (s4,s1,s2,s3)})
    ranked = sorted(test_dict.items(), key=operator.itemgetter(1))
    print(ranked)
    print("***")
    # print(ranked[:-10:-1])
    print(ranked[0:10])


if __name__ == '__main__':
    t0 = time()
    import data_grab
    df = pd.read_pickle('pickle_jar/review_text_sentiment_hierarchical_df')
    scores = ['score_lvl_1', 'score_lvl_2', 'score_lvl_3']

    # should be able to get rid of this in the next data grab
    df.previous_inspection_delta = df.previous_inspection_delta.fillna(0)
    df.previous_inspection_delta = df.previous_inspection_delta.dt.days
    df = make_bins(df)

    test2(df)

    print("{} seconds elapsed".format(time()-t0))
