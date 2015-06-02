import numpy as np
import pandas as pd
import data_grab
import text_processors
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier


# def get_data():
#     id_map = pd.read_csv("data/restaurant_ids_to_yelp_ids.csv")
#     id_dict = {}

#     # each Yelp ID may correspond to up to 4 Boston IDs
#     for i, row in id_map.iterrows():
#         boston_id = row["restaurant_id"]
#         # get the non-null Yelp IDs
#         non_null_mask = ~pd.isnull(row.ix[1:])
#         yelp_ids = row[1:][non_null_mask].values
#         for yelp_id in yelp_ids:
#             id_dict[yelp_id] = boston_id

#     with open("data/yelp_academic_dataset_review.json", 'r') as review_file:
#         # the file is not actually valid json since each line is an individual
#         # dict -- we will add brackets on the very beginning and ending in order
#         # to make this an array of dicts and join the array entries with commas
#         review_json = '[' + ','.join(review_file.readlines()) + ']'
#     # read in the json as a DataFrame
#     reviews = pd.read_json(review_json)

#     # drop columns that we won't use
#     reviews.drop(['review_id', 'type', 'user_id', 'votes'], 
#                  inplace=True, 
#                  axis=1)

#     # replace yelp business_id with boston restaurant_id
#     map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
#     reviews.business_id = reviews.business_id.map(map_to_boston_ids)

#     # rename first column to restaurant_id so we can join with boston data
#     reviews.columns = ["restaurant_id", "date", "stars", "text"]

#     # drop restaurants not found in boston data
#     reviews = reviews[pd.notnull(reviews.restaurant_id)]

#     train_labels = pd.read_csv("data/train_labels.csv", index_col=0)
#     submission = pd.read_csv("data/SubmissionFormat.csv", index_col=0)

#     return reviews, train_labels, submission


# def flatten_reviews(label_df, reviews):
#     """
#         label_df: inspection dataframe with date, restaurant_id
#         reviews: dataframe of reviews

#         Returns all of the text of reviews previous to each
#         inspection listed in label_df.
#     """
#     reviews_dictionary = {}
#     N = len(label_df)

#     for i, (pid, row) in enumerate(label_df.iterrows()):
#         # we want to only get reviews for this restaurant that ocurred before the inspection
#         pre_inspection_mask = (reviews.date < row.date) & (reviews.restaurant_id == row.restaurant_id)

#         # pre-inspection reviews
#         pre_inspection_reviews = reviews[pre_inspection_mask]

#         # join the text
#         all_text = ' '.join(pre_inspection_reviews.text)

#         # store in dictionary
#         reviews_dictionary[pid] = all_text

#         if i % 2500 == 0:
#             print '{} out of {}'.format(i, N)

#     # return series in same order as the original data frame
#     return pd.Series(reviews_dictionary)[label_df.index]


# def train_and_save(reviews, train_labels, submission):
#     train_text = flatten_reviews(train_labels, reviews)
#     with open('flattened_reviews_train_text.pkl', 'wb') as f:
#         pickle.dump(train_text, f)

#     test_text = flatten_reviews(submission, reviews)
#     with open('flattened_reviews_test_text.pkl', 'wb') as f:
#         pickle.dump(test_text, f)

#     return train_text, test_text


# def tfidf_and_save(train_text):
    # create a TfidfVectorizer object with english stop words
    # and a maximum of 1500 features (to ensure that we can
    # train the model in a reasonable amount of time)
    # vec = TfidfVectorizer(stop_words='english', max_features=5000)
    # with open('tfidf_vectorizer', 'wb') as f:
    #     pickle.dump(vec, f)

    # create the TfIdf feature matrix from the raw text
    # train_tfidf = vec.fit_transform(train_text)
    # with open('tfidf_array', 'wb') as f:
    #     pickle.dump(vec, f)

    # return train_tfidf


# def load_model(filename):
#     with open(filename) as f:
#         return pickle.dump(f)


def make_model(X, y):
    # # random forest object
    # model = RandomForestClassifier(n_estimators = 100)

    # create a Linear regresion object
    model = linear_model.LinearRegression()

    model.fit(X, y)
    return model


# reviews, train_labels, submission = get_data()

# # initial tfidf data grab


train_tfidf = text_processors.load_tfidf_matrix(params=5000)
train_labels, train_targets = data_grab.get_response()

model = make_model(train_tfidf, train_targets)
print model.score(train_tfidf, train_targets)
