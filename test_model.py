import pymongo

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from pprint import pprint
from progressbar import ProgressBar
from pymongo.cursor import CursorType
from sklearn.externals import joblib

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import BaggingClassifier

exhaust_cursor = pymongo.cursor.CursorType.EXHAUST
client = pymongo.MongoClient()
db = client.hygiene


def feature_the_whole_enchilada():
    # grabbing response and everything provided as a feature from database
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

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=3, verbose=1)
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


# set classifier to test
classifier = BaggingClassifier(n_estimators=100)

X, y = feature_text_only()

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', classifier)
    ])

scores = cross_val_score(pipeline, X, y, cv=5, n_jobs=-1, verbose=2)
print("Score of {} +/- {}").format(np.mean(scores), np.std(scores))

# save scores to csv
