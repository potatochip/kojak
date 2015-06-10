#!/usr/bin/env python

import csv
import logging
# import pymongo
import metrics
import data_grab
import sendMessage
import transformations
import text_processors
import visual_exploration

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from pprint import pprint
from datetime import datetime
from itertools import combinations
from prettytable import PrettyTable
from progressbar import ProgressBar
from scipy.sparse import coo_matrix, hstack
# from pymongo.cursor import CursorType

from sklearn.pipeline import Pipeline
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

n_jobs = 1  # -1 for full blast when not using computer


def logPrint(message):
    print(message)
    logger.info(message)


def extract_features(df):
    features = df.drop(['score_lvl_1', 'score_lvl_2', 'score_lvl_3', 'transformed_score'], axis=1)
    response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.float64)
    transformed_response = df['transformed_score']
    return features, response, transformed_response


def features_review_text():
    # # quick pipe - bypass the tfidf vectorizer
    tfidf_matrix = text_processors.load_tfidf_matrix(params=None)
    print("TFIDF matrix acquired.")
    train_labels, train_targets = data_grab.get_response()
    print("Targets loaded.")
    return tfidf_matrix, train_targets


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
    scores = cross_val_score(pipeline, X, y, cv=3, n_jobs=n_jobs, verbose=1)
    mean_score = np.mean(scores)
    std_dev_score = np.std(scores)
    logPrint("CV score of {} +/- {}".format(mean_score, std_dev_score))
    return mean_score, std_dev_score


def score_multiple(X, y, estimator_list, description):
    score_list = []
    for estimator in estimator_list:
        pipeline = Pipeline([
            # ('tfidf', TfidfVectorizer(stop_words='english')),
            # ('scaler', Normalizer()),
            ('est', estimator),
        ])
        logPrint("Scoring {}".format(str(pipeline)))
        logPrint('{} model scored'.format(description))
        mean_score, std_dev_score = score_model(X, y, pipeline)
        contest_score = contest_scoring(X, y, pipeline)
        dt = str(datetime.now())
        estimator_name = str(estimator).split('(')[0]
        score_list.append((dt, estimator_name, description, mean_score, std_dev_score, contest_score))
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
    model = pipeline.fit(X_train, y_train)
    score = contest_metric(model.predict(X_test), y_test)[0]
    logPrint("Contest score of {}".format(score))
    return score


def multi_feature_test(X_train, y_train, trans_list):
    combo_list = []
    for num in range(1, len(feature_list)+1):
        combo_list.extend([list(i) for i in combinations(feature_list, num)])

    for features in combo_list:
        description = '+'.join(features)+'/'+'+'.join(trans_list)
        test_models(X_train[features], y_train, description)


def test_models(X_train, y_train, description):
    if vectorized_docs:
        X_train = hstack([vectorized_docs[1], coo_matrix(X_train)])

    # score models
    score_list = score_multiple(X_train, y_train, estimator_list, description)
    x = PrettyTable(["Datetime", "Estimator", "Description", "Score Mean", "Score StndDev", "Contest Score"])
    x.padding_width = 1
    for i in score_list:
        x.add_row(i)
    print x
    print('\n')
    with open('models/estimator_log.csv', 'a') as f:
        writer = csv.writer(f, dialect='excel')
        # writer.writerow(["Datetime", "Estimator", "Description", "Score Mean", "Score StndDev", "Contest Score"])
        writer.writerows(score_list)


def transform(df, transformation):
    transformed = transformation(df)
    return transformed


def main():
    t0 = time()
    # train_df, test_df = data_grab.load_dataframes_selects(feature_list)
    # temp_feature_select = lambda x: x[['inspection_id', 'inspection_date', 'restaurant_id', 'time_delta', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3', 'transformed_score'].extend(feature_list)]
    # train_df = pd.read_pickle('models/training_df.pkl')
    # test_df = pd.read_pickle('models/test_df.pkl')
    train_df = data_grab.get_selects('train', feature_list)
    logPrint('dataframes retrieved')

    # transformations
    trans_list = []
    if transformation_list:
        for title, func in transformation_list:
            trans_list.append(title)
            print("Training set transform")
            train_df = transform(train_df, func)
    logPrint('feature extraction finished')

    X_train, y_train, transformed_y_train = extract_features(train_df)
    print X_train.head()
    if vectorized_docs:
        trans_list.append(vectorized_docs[0])

    multi_feature_test(X_train, y_train, trans_list)

    # make data exploration plots
    description = '_'.join(feature_list)+'_'+'_'.join(trans_list)
    visual_exploration.make_plots(X_train[feature_list], y_train, transformed_y_train, description)
    logPrint('plots made')

    logPrint('entire run finished')
    sendMessage.doneTextSend(t0, time(), 'test_model')



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
from sklearn.linear_model import SGDClassifier
# estimator_list = [LinearRegression(),
#                     MultinomialNB(),
#                     SGDClassifier()]

estimator_list = [LinearRegression()]


feature_list = ['time_delta', 'restaurant_id']
# feature_list = ['time_delta', 'review_text']

print("Grabbing vectorized docs")
vectorized_docs = text_processors.load_count_docs()
# vectorized_docs = None

# transformation_list = [('text_length', transformations.text_to_length)]
transformation_list = None

if __name__ == '__main__':
    main()
