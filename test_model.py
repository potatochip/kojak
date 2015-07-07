#!/usr/bin/env python

import csv
import scipy
import logging
# import pymongo
import socket
import metrics
import data_grab
import sendMessage
import transformations
import text_processors
import visual_exploration

import numpy as np
import pandas as pd

from time import time
from pprint import pprint
from datetime import datetime
from itertools import combinations
from prettytable import PrettyTable
from progressbar import ProgressBar
from scipy.sparse import coo_matrix, hstack
# from pymongo.cursor import CursorType

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split


LOG_FILENAME = 'test_model.log'
logging.basicConfig(filename=LOG_FILENAME, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# exhaust_cursor = pymongo.cursor.CursorType.EXHAUST
# client = pymongo.MongoClient()
# db = client.hygiene


if socket.gethostname() == 'ip-172-31-9-131':
    print('WORKING FULL BLAST')
    n_jobs = -1  # -1 for full blast when on aws
elif socket.gethostname() == 'trashcan.deathstar.private':
    print('BOOM!!!')
    n_jobs = -1
else:
    print('WORKING SLOW AS SHIT')
    n_jobs = 1  # since this funciton is not working on macbook air presently


def logPrint(message):
    print(message)
    logger.info(message)


def extract_features(df):
    features = df.drop(['score_lvl_1', 'score_lvl_2', 'score_lvl_3'], axis=1)
    response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.float64)  #for numerical progression
    # response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.int8)  # for categorical response
    return features, response


def griddy(X, y, pipeline):
    # get best params with cross fold validation for both the feature extraction and the classifier
    parameters = {
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        # sublinear tfidf vectorizer
        # 'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # words or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        # 'clf__alpha': (0.00001, 0.000001),
        # 'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=n_jobs, verbose=1)
    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, y)
    print "done in %0.3fs" % (time() - t0)
    print
    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])


def contest_metric(numpy_array_predictions, numpy_array_actual_values):
    return metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS)


def score_model(X, y, pipeline):
    # if isinstance(X, scipy.sparse.csr.csr_matrix) or isinstance(X, scipy.sparse.coo.coo_matrix):
    #     # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    #     model = pipeline
    #     model.fit(X, y)
    #     scores = accuracy_score(model.predict(X), y)
    # else:
    #     scores = cross_val_score(pipeline, X, y, cv=3, n_jobs=n_jobs, verbose=1)
    scores = cross_val_score(pipeline, X, y, cv=3, n_jobs=n_jobs, verbose=1)
    mean_score = np.mean(scores)
    std_dev_score = np.std(scores)
    logPrint("CV score of {} +/- {}".format(mean_score, std_dev_score))
    return mean_score


def score_multiple(X, y, estimator_list, description):
    # scores multiple estimators
    score_list = []
    for estimator in estimator_list:
        pipeline = Pipeline([
            # ('scaler', Normalizer()),
            ('est', estimator),
        ])
        logPrint('Scoring {} model on\n\t {}'.format(description, str(pipeline)))
        scores = []
        for i in y:
            single_y = i
            print("Scoring for response: {}".format(single_y))
            scores.append(score_model(X, y[single_y], pipeline))
        contest_score = contest_scoring(X, y, pipeline)
        dt = str(datetime.now())
        estimator_name = str(estimator).split('(')[0]
        row = [dt, estimator_name, description, np.mean(scores), contest_score]
        row[3:3] = scores
        score_list.append(row)
        print('\n')
    return score_list


def check_array_results(X, y, pipeline):
    # confirm that computing scores as an array yields the same results as computing them individually
    pass


def verify_no_negative(X, y, pipeline):
    # check whether scores are returned as negative and whether clipping required
    pass


def contest_scoring(X, y, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    s1 = pipeline.fit(X_train, y_train['score_lvl_1']).predict(X_test)
    s2 = pipeline.fit(X_train, y_train['score_lvl_2']).predict(X_test)
    s3 = pipeline.fit(X_train, y_train['score_lvl_3']).predict(X_test)
    results = np.dstack((s1, s2, s3))
    score = contest_metric(np.round(results[0]), np.array(y_test))
    logPrint("Contest score of {}".format(score))
    return score


def multi_feature_test(X_train, y_train, trans_list, combos=False):
    if combos:
        combo_list = []
        for num in range(1, len(feature_list)+1):
            combo_list.extend([list(i) for i in combinations(feature_list, num)])
    else:
        combo_list = [[i] for i in feature_list]

    for features in combo_list:
        description = '+'.join(features)+'/'+'+'.join(trans_list)
        test_models(X_train[features], y_train, description)


def test_models(X_train, y_train, description):
    global vectorized_docs

    if vectorized_docs and feature_list:
        X_train = hstack([vectorized_docs[1], coo_matrix(X_train)])
        logPrint('Matrices combined')
    elif vectorized_docs and not feature_list:
        X_train = vectorized_docs[1]
    elif not vectorized_docs and not feature_list:
        print('whoops!')
    elif not vectorized_docs and feature_list:
        pass

    # free up some memory
    vectorized_docs = None

    # score models
    score_list = score_multiple(X_train, y_train, estimator_list, description)
    x = PrettyTable(["Datetime", "Estimator", "Description", "Lvl 1 Accuracy", "Lvl 2 Accuracy", "Lvl 3 Accuracy", "Mean Accuracy", "Contest Metric"])
    x.padding_width = 1
    for i in score_list:
        x.add_row(i)
    print x
    print('\n')
    with open('predictions/estimator_log.csv', 'a') as f:
        writer = csv.writer(f, dialect='excel')
        # writer.writerow(["Datetime", "Estimator", "Description", "Lvl 1 Accuracy", "Lvl 2 Accuracy", "Lvl 3 Accuracy", "Mean Accuracy", "Contest Metric"])
        writer.writerows(score_list)


def transform(df, transformation):
    transformed = transformation(df)
    return transformed


def main():
    t0 = time()
    train_df = data_grab.get_selects('train', feature_list)
    logPrint('dataframes retrieved')

    # transformations
    trans_list = []
    if transformation_list:
        for title, func in transformation_list:
            trans_list.append(title)
            print("Training set transform")
            train_df = transform(train_df, func)
    X_train, y_train = extract_features(train_df)
    logPrint('feature extraction finished')

    # free up some memory
    del train_df

    if vectorized_docs:
        trans_list.append(vectorized_docs[0])

    if feature_list:
        # turn combos to True in order to try all combinatinos of features
        multi_feature_test(X_train, y_train, trans_list, combos=False)

        # make data exploration plots
        description = '_'.join(feature_list)+'_'+'_'.join(trans_list)
        visual_exploration.make_plots(X_train[feature_list], y_train, description)
        logPrint('plots made')
    else:
        test_models(X_train, y_train, 'single_feature_'+'_'.join(trans_list))

    sendMessage.doneTextSend(t0, time(), 'test_model')
    logPrint("Finished in {} seconds.".format(time() - t0))


# from sklearn.ensemble import BaggingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer

# set classifiers to test
# estimator = LinearRegression(Normalize=True)
# estimator = BaggingClassifier(n_estimators=100)
# estimator = RandomForestClassifier(n_estimators=100)

# # can use with text if convert X to dense with .toarray() but is super heavy on ram
# pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', estimator),
# ])

# estimator_list = [LinearRegression(),
#                     BaggingClassifier(),
#                     RandomForestClassifier(),
#                     MultinomialNB(),
#                     SGDClassifier()]

# pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english',
#                                                 max_df=0.8, max_features=200000, min_df=0.2,
#                                                 ngram_range=(1, 3), use_idf=True)),
#                     ('tfidf', TfidfTransformer()),
#                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                             alpha=1e-3, n_iter=5, random_state=42)),
#                     ])

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
# estimator_list = [
#                     # GaussianNB(),
#                     SGDClassifier()
#                     ]

estimator_list = [RandomForestClassifier()]


print("Grabbing vectorized docs")
feature_list = None
vectorized_docs = text_processors.load_tfidf_docs('train')
transformation_list = None

# feature_list = ['restaurant_stars', 'restaurant_attributes_accepts_credit_cards', 'user_votes_useful', 'restaurant_review_count']
# vectorized_docs = None
# transformation_list = [('text_length', transformations.text_to_length)]



if __name__ == '__main__':
    main()
