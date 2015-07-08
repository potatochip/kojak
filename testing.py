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

def contest_metric(numpy_array_predictions, numpy_array_actual_values):
    return metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS)

def contest_scoring(X, y, pipeline):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    s1 = pipeline.fit(X_train, y_train['score_lvl_1']).predict(X_test)
    s2 = pipeline.fit(X_train, y_train['score_lvl_2']).predict(X_test)
    s3 = pipeline.fit(X_train, y_train['score_lvl_3']).predict(X_test)
    results = np.dstack((s1, s2, s3))
    score = contest_metric(np.round(results[0]), np.array(y_test))
    print("Contest score of {}".format(score))
    return score

def score_model(X, y, pipeline):
    # multiprocessing depends on whether the size of the features are small enough for ram
    scores = cross_val_score(pipeline, X, y, cv=3, verbose=1, n_jobs=-1)
    mean_score = np.mean(scores)
    std_dev_score = np.std(scores)
    print("CV score of {} +/- {}".format(mean_score, std_dev_score))

def extract_features(df):
    features = df.drop(['score_lvl_1', 'score_lvl_2', 'score_lvl_3'], axis=1)
    response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.float64)  #for numerical progression
    # response = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.int8)  # for categorical response
    return features, response


if __name__ == '__main__':
    import data_grab
    train_df, test_df = data_grab.get_flats()
    X, y = extract_features(train_df)
    scores = y[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]

    tfidf = joblib.load( 'pickle_jar/tfidf_preprocessed_ngram3_sublinear_1mil')

    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.ensemble import BaggingClassifier

    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA

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
                ('clf', estimator),
        ])

        for score in scores:
            print(score)
            score_model(tfidf, y[score], pipeline)

        print
        contest_scoring(tfidf, y, pipeline)
