import logging
import pymongo
import metrics
import data_grab
import sendMessage
import text_processors

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from pprint import pprint
from progressbar import ProgressBar
from pymongo.cursor import CursorType
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


LOG_FILENAME = 'test_model.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
logger = logging.getLogger(__name__)

# exhaust_cursor = pymongo.cursor.CursorType.EXHAUST
# client = pymongo.MongoClient()
# db = client.hygiene

n_jobs = -1  # -1 for full blast when not using computer


def logPrint(message):
    print(message)
    logger.info(message)


def features_the_whole_enchilada():
    # grabbing everything
    target = db.target
    return features, response


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


def fit_and_submit(X_train, y_train, X_test, pipeline, filename):
    # predict the counts for the test set
    model =  pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    # clip the predictions so they are all greater than or equal to zero
    # since we can't have negative counts of violations
    predictions = np.clip(predictions, 0, np.inf)
    # write the submission file
    new_submission = data_grab.get_submission().copy()
    new_submission.iloc[:, -3:] = predictions.astype(int)
    new_submission.to_csv('predictions/'+filename)
    train_labels, train_targets = data_grab.get_response()
    print("Drivendata score of {}".format(contest_metric(predictions, train_targets)))


def make_feature_vis():
    pass


def contest_metric(numpy_array_predictions, numpy_array_actual_values):
    return metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS)


def score_model(X, y, pipeline):
    scores = cross_val_score(pipeline, X, y, cv=3, n_jobs=n_jobs, verbose=1)
    logPrint("Score of {} +/- {}").format(np.mean(scores), np.std(scores))


def score_multiple(X, y, estimator_list):
    for estimator in estimator_list:
        t0 = time()
        pipeline = Pipeline([
            # ('tfidf', TfidfVectorizer(stop_words='english')),
            # ('scaler', Normalizer()),
            ('est', estimator),
        ])
        logPrint("Scoring {}".format(str(pipeline)))
        score_model(X, y, pipeline)
        t1 = time()
        sendMessage.doneTextSend(t0, t1, estimator)


def score_single(X, y, estimator):
    pipeline = Pipeline([
        # ('tfidf', TfidfVectorizer(stop_words='english')),
        # ('scaler', Normalizer()),
        ('est', estimator),
    ])
    # # can use with text if convert X to dense with .toarray() but is super heavy on ram
    # pipeline = Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('clf', estimator),
    # ])
    score_model(X, y, pipeline)


# set classifiers to test
estimator = LinearRegression()
# estimator = BaggingClassifier(n_estimators=100)
# estimator = RandomForestClassifier(n_estimators=100)

estimator_list = [LinearRegression(normalize=True),
                  BaggingClassifier(),
                  RandomForestClassifier(),
                  MultinomialNB(),
                  SGDClassifier()]

t0 = time()
features, response = features_review_text()
print(features.shape)
# score_single(features, response, estimator)
# score_multiple(features, response, estimator_list)
t1 = time()
print("{} seconds elapsed.".format(int(t1 - t0)))

sendMessage.doneTextSend(t0, t1, 'test_model')

# # create submission file
train_text, test_text = data_grab.load_flattened_reviews()
fit_and_submit(test_text, estimator, 'ols_tfidf_None.csv')


# contest_metric()
# save scores to csv

# text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english',
#                                                 max_df=0.8, max_features=200000, min_df=0.2,
#                                                 ngram_range=(1, 3), use_idf=True)),
#                     ('tfidf', TfidfTransformer()),
#                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                             alpha=1e-3, n_iter=5, random_state=42)),
#                     ])
