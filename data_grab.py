from time import time

import numpy as np
import pandas as pd
import cPickle as pickle


# def get_reviews():
#     '''
#     returns reviews remapped to business_ids as a dataframe along with other review information
#     '''
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

#     # replace yelp business_id with boston restaurant_id
#     map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
#     reviews.business_id = reviews.business_id.map(map_to_boston_ids)

#     # rename first column to restaurant_id so we can join with boston data
#     reviews.columns = ["restaurant_id", "review_date", "review_id", "stars", "text", "type", "user_id", "votes"]

#     # drop restaurants not found in boston data
#     reviews = reviews[pd.notnull(reviews.restaurant_id)]

#     # unwind date and votes
#     reviews['review_year'] = reviews['review_date'].dt.year
#     reviews['review_month'] = reviews['review_date'].dt.month
#     reviews['review_day'] = reviews['review_date'].dt.day
#     reviews['vote_cool'] = [i[1]['cool'] for i in reviews.votes.iteritems()]
#     reviews['vote_funny'] = [i[1]['funny'] for i in reviews.votes.iteritems()]
#     reviews['vote_useful'] = [i[1]['useful'] for i in reviews.votes.iteritems()]

#     # drop redundant columns
#     return reviews


# def get_response():
#     train_labels = pd.read_csv("data/train_labels.csv", index_col=0)
#     # get just the targets from the training labels
#     train_targets = train_labels[['*', '**', '***']].astype(np.float64)
#     return train_labels, train_targets


def get_submission():
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=0)
    return submission


def get_reviews():
    with open("data/yelp_academic_dataset_review.json", 'r') as f:
        # the file is not actually valid json since each line is an individual
        # dict -- we will add brackets on the very beginning and ending in order
        # to make this an array of dicts and join the array entries with commas
        review_json = '[' + ','.join(f.readlines()) + ']'
    reviews = pd.read_json(review_json)
    return reviews


def get_tips():
    with open("data/yelp_academic_dataset_tip.json", 'r') as f:
        tip_json = '[' + ','.join(f.readlines()) + ']'
    tips = pd.read_json(tip_json)
    return tips


def get_users():
    with open("data/yelp_academic_dataset_user.json", 'r') as f:
        user_json = '[' + ','.join(f.readlines()) + ']'
    users = pd.read_json(user_json)
    users.columns = [u'user_average_stars', u'user_compliments', u'user_elite', u'user_fans', u'user_friends', u'user_name', u'user_review_count', u'user_type', u'user_id', u'user_votes', u'user_yelping_since']
    return users


def get_restaurants():
    with open("data/yelp_academic_dataset_business.json", 'r') as f:
        restaurant_json = '[' + ','.join(f.readlines()) + ']'
    restaurants = pd.read_json(restaurant_json)
    restaurants.columns = [u'restaurant_attributes', u'restaurant_id', u'restaurant_categories', u'restaurant_city', u'restaurant_full_address', u'restaurant_hours', u'restaurant_latitude', u'restaurant_longitude', u'restaurant_name', u'restaurant_neighborhoods', u'restaurant_open', u'restaurant_review_count', u'restaurant_stars', u'restaurant_state', u'restaurant_type']
    return restaurants


def get_checkins():
    with open("data/yelp_academic_dataset_checkin.json", 'r') as f:
        checkin_json = '[' + ','.join(f.readlines()) + ']'
    checkins = pd.read_json(checkin_json)
    checkins.columns = [u'restaurant_id', u'checkin_info', u'checkin_type']
    return checkins

def get_full_features():
    reviews = get_reviews()
    tips = get_tips()

    # some nan's will exist because of this. will exist where reviews columns and tips columns don't match up
    reviews_tips = reviews.append(tips)
    reviews_tips.columns = [u'restaurant_id', u'review_date', u'tip_likes', u'review_id', u'review_stars', u'review_text', u'review_type', u'user_id', u'review_votes']

    # saving this for tfidf vectorizer training later
    with open('models/reviews_tips_original_text.pkl', 'w') as f:
        pickle.dump(reviews_tips.review_text.tolist(), f)

    users = get_users()
    users_reviews_tips = pd.merge(reviews_tips, users, on='user_id')

    restaurants = get_restaurants()
    restaurants_users_reviews_tips = pd.merge(users_reviews_tips, restaurants, on='restaurant_id')

    # if checkins dont exist for a restaurant dont want to drop the restaurant values
    checkins = get_checkins()
    full_features = pd.merge(restaurants_users_reviews_tips, checkins, how='left', on='restaurant_id')

    id_map = pd.read_csv("data/restaurant_ids_to_yelp_ids.csv")
    id_dict = {}
    # each Yelp ID may correspond to up to 4 Boston IDs
    for i, row in id_map.iterrows():
        # get the Boston ID
        boston_id = row["restaurant_id"]
        # get the non-null Yelp IDs
        non_null_mask = ~pd.isnull(row.ix[1:])
        yelp_ids = row[1:][non_null_mask].values
        for yelp_id in yelp_ids:
            id_dict[yelp_id] = boston_id

    # replace yelp business_id with boston restaurant_id
    map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
    full_features.restaurant_id = full_features.restaurant_id.map(map_to_boston_ids)

    # drop restaurants not found in boston data
    full_features = full_features[pd.notnull(full_features.restaurant_id)]

    full_features.to_hdf('models/df_store.h5', 'full_features_df')
    return full_features


def load_full_features():
    df = pd.read_hdf('models/df_store.h5', 'full_features_df')
    return df


def transform_features(df):
    # create number representing days passed between inspection date and review date
    df['time_delta'] = (df.inspection_date - df.review_date).astype('timedelta64[D]')

    # remove columns with values without variance
    df.drop('restaurant_type', axis=1, inplace=True)

    scores = df[['*', '**', '***']].astype(np.float64)
    df['transformed_score'] = scores.multiply([1,3,5], axis=1).sum(axis=1)

    # df['vote_cool'] = [i[1]['cool'] for i in df.votes.iteritems()]
    # df['vote_funny'] = [i[1]['funny'] for i in df.votes.iteritems()]
    # df['vote_useful'] = [i[1]['useful'] for i in df.votes.iteritems()]
    # df['review_year'] = df['review_date'].dt.year
    # df['review_month'] = df['review_date'].dt.month
    # df['review_day'] = df['review_date'].dt.day
    # df['inspection_dayofweek'] = df.inspection_date.dt.dayofweek
    # df['inspection_quarter'] = df.inspection_date.dt.quarter
    # df['inspection_dayofyear'] = df.inspection_date.dt.dayofyear
    # df['inspection_weekofyear'] = df.inspection_date.dt.weekofyear

    return df


def make_feature_response(feature_df, response_df):
    # convert date to datetime object
    response_df.inspection_date = pd.to_datetime(pd.Series(response_df.inspection_date))
    # combine features and response
    features_response = pd.merge(feature_df, response_df, on='restaurant_id')

    # remove reviews and tips that occur after an inspection - canceled because then end up removing restaurants that trying to predict for. future reviews still have predictive power and frames with no reviews but other information still have predictive power
    # no_future = features_response[features_response.review_date < features_response.inspection_date]

    return features_response


def make_train_test():
    full_features = get_full_features()
    training_response = pd.read_csv("data/train_labels.csv", index_col=None)
    training_response.columns = ['inspection_id', 'inspection_date', 'restaurant_id', '*', '**', '***']
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=None)
    submission.columns = ['inspection_id', 'inspection_date', 'restaurant_id', '*', '**', '***']
    # combine features and response
    training_df = make_feature_response(full_features, training_response)
    test_df = make_feature_response(full_features, submission)

    # transform dataframes
    transformed_training_df = transform_features(training_df)
    transformed_test_df = transform_features(test_df)

    # convert restaurant_id's into numbers representing the restaurant then applying the same
    # categories to the submission dataframe
    restaurant_categories = pd.Categorical.from_array(transformed_training_df.restaurant_id)
    transformed_training_df['restaurant_id_number'] = restaurant_categories.codes
    transformed_test_df['restaurant_id_number'] = restaurant_categories.categories.get_indexer(transformed_test_df.restaurant_id)

    # save dataframes
    transformed_training_df.to_hdf('models/df_store.h5', 'transformed_training_df')
    transformed_test_df.to_hdf('models/df_store.h5', 'transformed_test_df')
    return transformed_training_df, transformed_test_df


def load_train_df():
    train_df = pd.read_hdf('models/df_store.h5', 'transformed_training_df')
    return train_df


def load_test_df():
    test_df = pd.read_hdf('models/df_store.h5', 'transformed_test_df')
    return test_df


def load_dataframes():
    # test refers to what is being submitted to the competition
    # running cross_val_score on train only
    train_df = load_train_df()
    test_df = load_test_df()
    return train_df, test_df


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
#     test_text = flatten_reviews(submission, reviews)
#     joblib.dump(train_text, 'models/flattened_train_reviews')
#     joblib.dump(test_text, 'models/flattened_test_reviews')
#     return train_text, test_text


# def load_flattened_reviews():
#     train_text = joblib.load('models/flattened_train_reviews.pkl')
#     test_text = joblib.load('models/flattened_test_reviews.pkl')
#     return train_text, test_text


def main():
    t0 = time()
    # reviews = get_reviews()
    # train_labels, train_targets = get_response()
    # submission = get_submission()
    # train_and_save(reviews, train_labels, submission)

    make_train_test()
    print("{} seconds elapsed.".format(time() - t0))


if __name__ == '__main__':
    main()
