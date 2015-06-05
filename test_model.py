#!/usr/bin/env python

import logging
import pymongo
import metrics
import data_grab
import sendMessage
import transformations
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

n_jobs = 1  # -1 for full blast when not using computer


def logPrint(message):
    print(message)
    logger.info(message)


def logTime(t0, t1, description):
    logPrint("{} seconds elapsed from start after {}.".format(int(t1 - t0), description))


def extract_features(df, features):
    features = df[features]
    response = df[['*', '**', '***']].astype(np.float64)
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


def fit_and_submit(X_train, y_train, test_df, pipeline, filename):
    X_test, y_test = extract_features(test_df, features)

    # predict the counts for the test set
    model = pipeline.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # clip the predictions so they are all greater than or equal to zero
    # since we can't have negative counts of violations
    # SHOULD TRY CLIPPING AFTER AVERAGING SCORES ALSO
    predictions = np.clip(predictions, 0, np.inf)

    # averaging by mean, SHOULD TRY ALT METHODS OF GROUPING SCORES TOGETHER
    test_df[['*', '**', '***']] = predictions
    submission_scores = test_df.groupby(['restaurant_id', 'inspection_date', 'inspection_id'])['*', '**', '***'].mean()
    temp = submission_scores.reset_index().set_index('inspection_id')
    indexed = temp.reindex(new_submission.index)

    # write the submission file
    new_submission = data_grab.get_submission()
    if new_submission.shape != indexed.shape:
        logPrint("ERROR: Submission and prediction have different shapes")
    new_submission.iloc[:, -3:] = np.round(indexed[['*', '**', '***']]).astype(int)
    new_submission.to_csv('predictions/'+filename)

    # print("Drivendata score of {}".format(contest_metric(predictions, train_targets)))


def contest_metric(numpy_array_predictions, numpy_array_actual_values):
    return metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS)


def score_model(X, y, pipeline):
    scores = cross_val_score(pipeline, X, y, cv=3, n_jobs=n_jobs, verbose=1)
    logPrint("Score of {} +/- {}".format(np.mean(scores), np.std(scores)))


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


def make_plots(X, y, description):
    plt.rcParams["figure.figsize"] = (10, 8)

    # weigh y according to competition weights and sum
    trans_y = pd.DataFrame(y.multiply([1,3,5], axis=1).sum(axis=1), columns=['transformed_score'])
    data = pd.concat([X, y], axis=1)
    trans_data = pd.concat([X, trans_y], axis=1)

    feature_hist = X.hist(bins=100)
    feature_hist.savefig('graphs/feature_histogram_'+description)
    response_hist = y.hist(bins=50)
    response_hist.savefig('graphs/response_histogram')
    trans_y_hist = trans_y.hist(bins=100)
    trans_y_hist.savefig('graphs/transformed_response_histogram_')

    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF",
                              "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
    sns.corrplot(data, annot=False, diag_names=False, cmap=cmap)
    ax.grid(False)
    ax.savefig('graphs/correlation_plot_'+description)



# set classifiers to test
estimator = LinearRegression()
# estimator = LinearRegression(Normalize=True)
# estimator = BaggingClassifier(n_estimators=100)
# estimator = RandomForestClassifier(n_estimators=100)

# # can use with text if convert X to dense with .toarray() but is super heavy on ram
# pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('clf', estimator),
# ])

estimator_list = [LinearRegression(),
                  BaggingClassifier(),
                  RandomForestClassifier(),
                  MultinomialNB(),
                  SGDClassifier()]

t0 = time()

features = ['time_delta', 'review_text']

train_df, test_df = data_grab.load_dataframes()
logTime(t0, time(), 'dataframes retrieved')

X_train, y_train = extract_features(train_df, features)
X_train = transformations.text_to_length(X_train)
test_df = transformations.text_to_length(test_df)
logTime(t0, time(), 'feature extraction')

# score models
# score_model(X_train, y_train, estimator)
# score_multiple(X_train, y_train, estimator_list)
# logTime(t0, time(), 'model(s) scored')

# # make data exploration plots
# make_plots(X_train, y_train)
# logTime(t0, time(), 'plots made')

# # make submission file
fit_and_submit(X_train, y_train, test_df, estimator, 'ols.csv')

logTime(t0, time(), 'entire run')
sendMessage.doneTextSend(t0, time(), 'test_model')


# contest_metric()


# text_clf = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, stop_words='english',
#                                                 max_df=0.8, max_features=200000, min_df=0.2,
#                                                 ngram_range=(1, 3), use_idf=True)),
#                     ('tfidf', TfidfTransformer()),
#                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
#                                             alpha=1e-3, n_iter=5, random_state=42)),
#                     ])
