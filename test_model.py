import pymongo
import metrics
import data_grab
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

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


exhaust_cursor = pymongo.cursor.CursorType.EXHAUST
client = pymongo.MongoClient()
db = client.hygiene

n_jobs = -1  # -1 for full blast when not using computer


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(counter=0)
def update_pbar():
    update_pbar.counter += 1
    pbar.update(update_pbar.counter)


def feature_the_whole_enchilada():
    # grabbing response and everything provided as a feature from database
    target = db.target
    return features, response


def feature_review_text():
    target = db.target
    return features, response


def griddy(pipeline):
    # get best params with cross fold validation for both the feature extraction and the classifier
    parameters = {
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
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


def fit_and_submit():
    pass


def make_feature_vis():
    pass


def contest_metric():
    print(metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS))


def score_model(X, y, model=None, pipe='text'):
    # set classifier to test
    if model == 'ols':
        classifier = LinearRegression()
    if model == 'bagging':
        classifier = BaggingClassifier(n_estimators=100)
    if model == 'forest':
        classifier = RandomForestClassifier(n_estimators=100)
    if not model:
        print('no model specified. error inbound.')

    if pipe == 'text':
        pipeline = Pipeline([
                # ('scaler', Normalizer()),
                ('clf', classifier),
            ])

    if pipe == 'numeric':
        # can use with text if convert X to dense with .toarray() but is super heavy on ram
        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', classifier),
            ])

    scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=n_jobs, verbose=1)
    print("Score of {} +/- {}").format(np.mean(scores), np.std(scores))


t0 = time()
# train_text, test_text = data_grab.load_flattened_reviews()
train_tfidf = text_processors.load_tfidf_matrix()
print(train_tfidf.shape)
# vec, train_tfidf = text_processors.tfidf_and_save(train_text)
train_labels, train_targets = data_grab.get_response()
score_model(train_tfidf.toarray(), train_targets, 'ols', 'text')
print("{} seconds elapsed.".format(time() - t0))

# contest_metric()

# save scores to csv
