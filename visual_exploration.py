import data_grab
import sendMessage
import matplotlib
matplotlib.use('Agg') 


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 8)


def distributions(variable, filename):
    # better with bins=50 for the y.hist
    variable.hist(bins=100)
    plt.savefig('visuals/'+filename)
    print('visuals/'+filename)
    plt.close()


def interactions(X, y, description):
    for i in X.iteritems():
        feature_title = i[0]
        plt.plot(i[1], y, '.', alpha=.4)
        plt.savefig('visuals/interact_'+feature_title+'_'+description)
        print('visuals/interact_'+feature_title+'_'+description)
        plt.close()


def coefficients(X, y, y_formula):
    X_title = "_".join([i for i in X.columns.tolist()])
    X_formula = " * ".join([i for i in X.columns.tolist()])
    formula = y_formula + " ~ " + X_formula
    print("\n* Formula: {}".format(formula))
    data = pd.concat([X, y], axis=1)
    g = sns.coefplot(formula, data, intercept=True)
    g.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig('visuals/coefficient_'+X_title+'_'+y_formula)
    print('visuals/coefficient_'+X_title+'_'+y_formula)
    plt.close()


def correlations(data, X):
    X_title = "_".join([i for i in X.columns.tolist()])
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.blend_palette(["#00008B", "#6A5ACD", "#F0F8FF",
                              "#FFE6F8", "#C71585", "#8B0000"], as_cmap=True)
    sns.corrplot(data, annot=False, diag_names=False, cmap=cmap)
    ax.grid(False)
    plt.savefig('visuals/correlation_'+X_title)
    print('visuals/correlation_'+X_title)
    plt.close()


def response_histograms(data):
    distributions(data[['score_lvl_1', 'score_lvl_2', 'score_lvl_3', 'transformed_score']], 'all_response_histogram')


def strip(X, y, description):
    for i in X.iteritems():
        feature_title = i[0]
        sns.stripplot(x=i[1], y=y, jitter=True)
        plt.savefig('visuals/strips_'+feature_title+'_'+description)
        print('visuals/correlation_'+feature_title+'_'+description)
        plt.close()


def make_plots(X, y, description):
    # weigh scores according to competition weights and sum
    scores = y[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
    transformed_y = pd.DataFrame(scores.multiply([1, 3, 5], axis=1).sum(axis=1), columns=['transformed_score'])
    # transformed_y = pd.DataFrame(scores.apply(lambda x: np.average(x, weights=[1, 3, 5], axis=1)), columns=['transformed_score'])

    data = pd.concat([X, y], axis=1)
    # transformed_data = pd.concat([X, transformed_y], axis=1)

    response_histograms(data.join(transformed_y))

    # features histograms
    distributions(X, 'histograms_combined_'+description)

    # histograms and interaction plots for each feature
    interactions(X, transformed_y, 'transformed_y')
    interactions(X, y['score_lvl_1'], 'score_lvl_1')
    interactions(X, y['score_lvl_2'], 'score_lvl_2')
    interactions(X, y['score_lvl_3'], 'score_lvl_3')

    # coefficient plots for all features across transformed score and each individual score
    coefficients(X, transformed_y, 'transformed_score')
    coefficients(X, y['score_lvl_1'], 'score_lvl_1')
    coefficients(X, y['score_lvl_2'], 'score_lvl_2')
    coefficients(X, y['score_lvl_3'], 'score_lvl_3')

    # strip plots
    strip(X, transformed_y, 'transformed_y')
    strip(X, y['score_lvl_1'], 'score_lvl_1')
    strip(X, y['score_lvl_2'], 'score_lvl_2')
    strip(X, y['score_lvl_3'], 'score_lvl_3')

    # feature correlation plot
    correlations(data, X)
