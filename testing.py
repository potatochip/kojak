import numpy as np
import pandas as pd
import data_grab
from pandas.io.json import json_normalize
from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
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
from pprint import pprint
import operator
import sendMessage
from scipy.sparse import csr_matrix, hstack
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, DictionaryLearning, FactorAnalysis, FastICA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import cross_val_score

from sklearn.decomposition import PCA, TruncatedSVD

import logging
LOG_FILENAME = 'testing.log'
logging.basicConfig(filename=LOG_FILENAME, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def logPrint(message):
    print(message)
    logger.info(message)

def contest_metric(numpy_array_predictions, numpy_array_actual_values):
    return metrics.weighted_rmsle(numpy_array_predictions, numpy_array_actual_values,
            weights=metrics.KEEPING_IT_CLEAN_WEIGHTS)

def raw_scoring(X, y, pipeline, rs=42):
    '''since cross_val_score doesn't allow you to round the results beforehand'''
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=rs)

    p1 = pipeline.fit(xtrain, ytrain['score_lvl_1']).predict(xtest)
    r1 = np.clip(np.round(p1), 0, np.inf)
    score1 = accuracy_score(ytest['score_lvl_1'], r1)
    logPrint("Level 1 accuracy score of {}".format(score1))
    p2 = pipeline.fit(xtrain, ytrain['score_lvl_2']).predict(xtest)
    r2 = np.clip(np.round(p2), 0, np.inf)
    score2 = accuracy_score(ytest['score_lvl_2'], r2)
    logPrint("Level 2 accuracy score of {}".format(score2))
    p3 = pipeline.fit(xtrain, ytrain['score_lvl_3']).predict(xtest)
    r3 = np.clip(np.round(p3), 0, np.inf)
    score3 = accuracy_score(ytest['score_lvl_3'], r3)
    logPrint("Level 3 accuracy score of {}".format(score3))

    results = np.dstack((p1, p2, p3))[0]
    rounded = np.clip(np.round(results), 0, np.inf)
    score = contest_metric(rounded, np.array(ytest))
    # logPrint("Contest score of {}".format(score))
    if score < 1.282:
        logPrint("Contest score of {}".format(score))


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
        logPrint("Level 1 accuracy score of {} for {}".format(score, features))
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

def log_it(x):
#     second_min = x[x != x.min()].min()
#     x.replace(x.min(), second_min/2, inplace=True)
#     return np.log(x)
    x = x + 1
    return np.log(x)

def pivot_test(levels):
    pool = Pool()
    score_list = pool.map(pivot_pool, range(1,levels))
    pool.close()
    pool.join()
    for i in ['lsa', 'pca']:
        global pv_df
        global pv_feature
        global pv_score
        global y
        X = pivot_feature(pv_df, pv_feature, limit=None, decomp=i)
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42)
        p = RandomForestClassifier(n_jobs=-1, random_state=42).fit(xtrain, ytrain[pv_score]).predict(xtest)
        score = accuracy_score(ytest[pv_score], np.clip(np.round(p), 0, np.inf))
        score_list.append(score)
    return score_list

def pivot_pool(i):
    X = pivot_feature(pv_df, pv_feature, limit=i, decomp=None)
    global y
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state=42)
    p = RandomForestClassifier(n_jobs=-1, random_state=42).fit(xtrain, ytrain[pv_score]).predict(xtest)
    score = accuracy_score(ytest[pv_score], np.clip(np.round(p), 0, np.inf))
    return score

def sim_length(x):
    try:
        return len(np.where(x > 0.6)[0])
    except:
        return x

def pivot_feature(df, feature, limit=None, decomp='lsi', decomp_features=2, fill='median'):
    # make the large dataframe faster to handle on pivot
    temp = df[['inspection_id', 'enumerated_review_delta'] + [feature]]

    # pivot so that each inspection id only has one observation with each review a feature for that observation
    pivoted_feature = temp.pivot('inspection_id', 'enumerated_review_delta')[feature]


    # pivoting creates a number of empty variables when they have less than the max number of reviews
    if fill == 'median':
        fill_empties = lambda x: x.fillna(x.median())
    elif fill == 'mean':
        fill_empties = lambda x: x.fillna(x.mean())
    elif fill == 0:
        fill_empties = lambda x: x.fillna(0)
    elif fill == 'inter':
        fill_empties = lambda x: x.interpolate()
    elif fill == None:
        fill_empties = lambda x: x
    else:
        raise Exception

    pivoted_feature = pivoted_feature.apply(fill_empties, axis=1)

    if decomp == 'lsi':
        decomposition = TruncatedSVD(decomp_features)
    elif decomp == 'pca':
        decomposition = PCA(decomp_features, whiten=True)
    elif decomp == 'kpca':
        decomposition = KernelPCA(decomp_features)
    elif decomp == 'dict':
        decomposition = DictionaryLearning(decomp_features)
    elif decomp == 'factor':
        decomposition = FactorAnalysis(decomp_features)
    elif decomp == 'ica':
        decomposition = FastICA(decomp_features)
    elif decomp == None:
        pass
    else:
        raise Exception

    if not limit:
        try:
            return decomposition.fit_transform(pivoted_feature)
        except:
            return pivoted_feature
    else:
        try:
#             return decomposition.fit_transform(pivoted_feature)[: , 0:limit]
            return decomposition.fit_transform(pivoted_feature[[i for i in range(limit)]])
        except:
            return pivoted_feature[[i for i in range(limit)]]

def test1():
    '''testing multiple models'''
    # X, y = extract_features(df)

    X = joblib.load('pickle_jar/test_matrix')
    y = joblib.load('pickle_jar/final_y')
    print(X.shape)
    print(y.shape)

    # combo = pd.read_pickle('pickle_jar/pre-pivot_all_review_combo_365')
    # df = pd.read_pickle('pickle_jar/pre-pivot_365')
    # df = pd.read_pickle('pickle_jar/pre-pivot_all_non_review')
    #
    # tfidf = joblib.load('pickle_jar/tfidf_preprocessed_ngram3_sublinear_1mil_pivot')
    #
    # scores = ['score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    # y = df[scores]
    #
    # from sklearn.decomposition import TruncatedSVD
    # lsa = TruncatedSVD(100)
    # lsa_tfidf = lsa.fit_transform(tfidf)

    # X = hstack([tfidf, log_it(df[['previous_inspection_delta']])])
    # X = np.concatenate([pd.DataFrame(log_it(df.previous_inspection_delta)), lsa_tfidf], axis=1)

    # set classifiers to test
    estimator_list = [
            # SGDClassifier(n_jobs=-1, random_state=42),
            # Perceptron(n_jobs=-1, random_state=42),  # gets some nuances
            # SGDRegressor(random_state=42),
            # KNeighborsRegressor(),  # gets some nuances
            RandomForestClassifier(n_jobs=-1, random_state=42),
            RandomForestRegressor(random_state=42),
            # LinearRegression(),# gets some nuances

        ]

    for estimator in estimator_list:
        t0 = time()
        logPrint(estimator)
        pipeline = Pipeline([
                # ('zero_variance_removal', VarianceThreshold()),
                # ('k_best', SelectKBest(score_func=f_classif, k=20)),
                # ('no_negative', MinMaxScaler()),
                # ('normalizer', Normalizer(norm='l2')), #  for text classification and clustering
                # ('normalizer', Normalizer(copy=False)),
                # ('scaler', StandardScaler()),
                # ('scaler', StandardScaler(with_mean=False)), #  for sparse matrix
                ('clf', estimator),
        ])
        raw_scoring(X, y, pipeline, rs=7)
        logPrint('\n')

        sendMessage.doneTextSend(t0, time(), 'multiple models')



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
    logPrint(ranked)
    logPrint("***")
    logPrint(ranked[:-20:-1])
    # logPrint(ranked[0:20])


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
        logPrint("bin size: {}".format(i))
        s1, s2, s3, s4, c = raw_scoring(X, y, pipeline)
        test_dict.update({i: (s4,s1,s2,s3)})
    ranked = sorted(test_dict.items(), key=operator.itemgetter(1))
    logPrint(ranked)
    logPrint("***")
    # logPrint(ranked[:-10:-1])
    logPrint(ranked[0:10])

def test4():
    '''get best params with cross fold validation for both the feature extraction and the classifier'''
    tstart = time()
    X = joblib.load('pickle_jar/final_matrix')
    y = joblib.load('pickle_jar/final_y')

    y = y.score_lvl_1

    parameters = {
                #   "max_depth": [3, None],
                #   "max_features": [1, 3, 10],
                #   "min_samples_split": [1, 3, 10],
                #   "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    clf = RandomForestClassifier(n_jobs=-1)
    grid_search = GridSearchCV(clf, parameters, verbose=5)
    print "Performing grid search..."
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

    t1 = time()
    sendMessage.doneTextSend(tstart, t1, 'text_processors')


def test5():
    '''crossvalscore on rf full matrix just to confirm'''
    X = joblib.load('pickle_jar/final_matrix')
    y = joblib.load('pickle_jar/final_y')

    score_list = ['score_lvl_1', 'score_lvl_2', 'score_lvl_3']

    clf = RandomForestClassifier(n_jobs=-1, random_state=42)

    for i in score_list:
        # kf = KFold(len(y[i]), shuffle=True)
        skf = StratifiedKFold(y[i], shuffle=True, n_folds=10)
        scores = cross_val_score(clf, X, y[i], cv=skf, verbose=5,)
        logPrint("Working on {}".format(i))
        loglogPrint("{} scores: {}".format(i, scores))
        loglogPrint("CV score of {} +/- {}".format(np.mean(scores), np.std(scores)))
        loglogPrint('\n')

def test6():
    combo = pd.read_pickle('pickle_jar/pre-pivot_all_review_combo_365')
    df = pd.read_pickle('pickle_jar/post-pivot_non_review_data_365')

    sim_length_list = []
    topics = ['manager', 'supervisor', 'training', 'safety', 'disease', 'ill', 'sick', 'poisoning', 'poison', 'hygiene', 'raw', 'undercooked', 'cold', 'clean', 'sanitary', 'wash', 'jaundice', 'yellow', 'hazard', 'inspection', 'violation', 'gloves', 'hairnet', 'nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'bacteria', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold']
    pbar = ProgressBar(maxval=len(topics)).start()
    for index, i in enumerate(topics):
        combo[i] = combo[i].apply(sim_length)
    #     sim_length_list.append(combo[i].apply(sim_length).tolist())
        pbar.update(index)
    pbar.finish()

    flist = ['nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'bacteria', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold']
    global pv_df
    global pv_score
    pv_df = combo
    # pv_feature = 'nails'
    pv_score = 'score_lvl_1'
    y = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
    for i in flist:
        global pv_feature
        pv_feature = i
        score_list = pivot_test(100)
        indexed = np.argsort(score_list)
        logPrint("Best level of {} with a score of {} for {}".format(indexed[-1] + 1, np.round(score_list[indexed[-1]], 4), i))


def test7():
    combo = pd.read_pickle('pickle_jar/pre-pivot_all_review_combo_365')
    df = pd.read_pickle('pickle_jar/post-pivot_non_review_data_365')
    tfidf_pivot = pd.read_pickle('pickle_jar/lsa_tfidf_2mil_pivot_365_100c')

    pipeline = Pipeline([
                        ('zero_variance_removal', VarianceThreshold()),
                        ('clf', RandomForestClassifier(n_jobs=-1, random_state=42)),
                        ])

    y = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]

    test_list = ['restaurant_street', 'restaurant_zipcode',  'inspection_year', 'inspection_month', 'inspection_day', 'inspection_dayofweek', 'inspection_quarter']

    X = pivot_feature(combo, feature='polarity', limit=None, decomp='factor')
    X = np.hstack((X, tfidf_pivot.groupby('inspection_id').mean()))
    f = open('feature_output.txt', 'w')
    score_1_features = []
    score_2_features = []
    score_3_features = []
    comp_score_features = []
    for i in test_list:
        logPrint('testing {} as categorical'.format(i))
        try:
            dummies = pd.get_dummies(df[[i]])
            newX = np.hstack((X, dummies)) # as categorical
            s1, s2, s3, cscore, c = raw_scoring(newX, y, pipeline, rs=42)
            if s1 > .4018: score_1_features.append(i)
            if s2 > .7816: score_2_features.append(i)
            if s3 > .6604: score_3_features.append(i)
            if cscore < 1.282: comp_score_features.append(i)
            for j in dummies.iteritems():
                logPrint('in {} testing individual dummy {}'.format(i, j[0]))
                newX = np.hstack((X, pd.DataFrame(j[1])))
                s1, s2, s3, cscore, c = raw_scoring(newX, y, pipeline, rs=42)
                if s1 > .4018: score_1_features.append(j[0])
                if s2 > .7816: score_2_features.append(j[0])
                if s3 > .6604: score_3_features.append(j[0])
                if cscore < 1.282: comp_score_features.append(j[0])
                # logPrint('log_it version')
                # newX = np.hstack((X, log_it(pd.DataFrame(j[1]))))
                # s1, s2, s3, cscore, c = raw_scoring(newX, y, pipeline, rs=42)
                if s1 > .4018: score_1_features.append(j[0])
                if s2 > .7816: score_2_features.append(j[0])
                if s3 > .6604: score_3_features.append(j[0])
                if cscore < 1.282: comp_score_features.append(j[0])
        except Exception as e:
                logPrint('categorical failed')
                logPrint(e)
        logPrint('testing {} as numerical')
        try:
            newX = np.hstack((X, df[[i]]))
            s1, s2, s3, cscore, c = raw_scoring(newX, y, pipeline, rs=42)
            if s1 > .4018: score_1_features.append(i)
            if s2 > .7816: score_2_features.append(i)
            if s3 > .6604: score_3_features.append(i)
            if cscore < 1.282: comp_score_features.append(i)
            # logPrint('log_it version')
            # newX = np.hstack((X, log_it(df[[i]])))
            # s1, s2, s3, cscore, c = raw_scoring(newX, y, pipeline, rs=42)
            if s1 > .4018: score_1_features.append(i)
            if s2 > .7816: score_2_features.append(i)
            if s3 > .6604: score_3_features.append(i)
            if cscore < 1.282: comp_score_features.append(i)
        except Exception as e:
            logPrint('numerical failed')
            logPrint(e)
        logPrint('\n')
    f.write('score_lvl_1\n')
    for i in set(score_1_features):
        f.write("'{}',\n".format(i))
    f.write('\nscore_lvl_2\n')
    for i in set(score_2_features):
        f.write("'{}',\n".format(i))
    f.write('\nscore_lvl_3\n')
    for i in set(score_3_features):
        f.write("'{}',\n".format(i))
    f.write('\ncompetition_score\n')
    for i in set(comp_score_features):
        f.write("'{}',\n".format(i))
    f.close()

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class RandomForestRegressorWithCoef(RandomForestRegressor):
    def fit(self, *args, **kwargs):
        super(RandomForestRegressorWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

def test8():
    from sklearn.feature_selection import RFECV

    final = pd.read_pickle('pickle_jar/final_no_tfidf_or_restaurant_id')
    tfidf_pivot = pd.read_pickle('pickle_jar/lsa_tfidf_2mil_pivot_365_100c')
    # tfidf = joblib.load('pickle_jar/full_post_pivot_tfidf_lsa')
    df = pd.read_pickle('pickle_jar/post-pivot_non_review_data_365')

    y = final[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
    X = final.drop(['score_lvl_1', 'score_lvl_2', 'score_lvl_3', 'inspection_id'], axis=1)
    X = np.hstack((X, pd.get_dummies(df.restaurant_id)))
    X = np.hstack((X, tfidf_pivot.groupby('inspection_id').mean()))
    # X = np.hstack((X, tfidf))

    print(X.shape)
    # est = RandomForestClassifierWithCoef(n_jobs=-1)
    est = RandomForestRegressorWithCoef(n_jobs=-1)
    score_list = ['score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    for i in score_list:
        skf = StratifiedKFold(y[i], shuffle=True, n_folds=3)
        rfecv = RFECV(estimator=est, cv=skf, verbose=5, step=.01, scoring='mean_squared_error')
        Xnew = rfecv.fit_transform(X, y[i])
        # joblib.dump(rfecv, 'pickle_jar/rfecv_restid_tfidf_pivot_RFR_'+i)
        joblib.dump(Xnew, 'pickle_jar/Xnew_restid_tfidf_pivot_RFR_'+i)

#rfecv step is 1 for rfc. for rfr it is .01

if __name__ == '__main__':
    t0 = time()

    # import data_grab
    # df = pd.read_pickle('pickle_jar/review_text_sentiment_hierarchical_df')
    # scores = ['score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    #
    # df = make_bins(df)

    #testing with cv folds 10
    # test5()

    # testing full matrix combined with full tfidf randomforest
    # test5()

    # test8()
    test8()

    logPrint("{} seconds elapsed".format(time()-t0))
