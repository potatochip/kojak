'''
must use python3. cant store unicode in hdf5 in python2
'''

from time import time

import numpy as np
import pandas as pd
import cPickle as pickle
import unicodedata
from pandas.io.json import json_normalize
import sendMessage
import json
import re


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

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input


def get_submission():
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=0)
    return submission


def get_reviews():
    data = []
    with open("data/yelp_academic_dataset_review.json", 'r') as f:
        for line in f:
            # reads text from json as str rather than unicode as json module does
            data.append(byteify(json.loads(line)))
    reviews = json_normalize(data)
    return reviews


def get_tips():
    data = []
    with open("data/yelp_academic_dataset_tip.json", 'r') as f:
        for line in f:
            data.append(byteify(json.loads(line)))
    tips = json_normalize(data)
    return tips


def get_users():
    data = []
    with open("data/yelp_academic_dataset_user.json", 'r') as f:
        for line in f:
            data.append(byteify(json.loads(line)))
    users = json_normalize(data)
    users.columns = ['user_average_stars', 'user_compliments.cool', 'user_compliments.cute', 'user_compliments.funny', 'user_compliments.hot', 'user_compliments.list', 'user_compliments.more', 'user_compliments.note', 'user_compliments.photos', 'user_compliments.plain', 'user_compliments.profile', 'user_compliments.writer', 'user_elite', 'user_fans', 'user_friends', 'user_name', 'user_review_count', 'user_type', 'user_id', 'user_votes.cool', 'user_votes.funny', 'user_votes.useful', 'user_yelping_since']
    return users


def get_restaurants():
    data = []
    with open("data/yelp_academic_dataset_business.json", 'r') as f:
        for line in f:
            data.append(byteify(json.loads(line)))
    restaurants = json_normalize(data)
    # duplicate columns exist for 'good for kids'
    shorter = restaurants['attributes.Good For Kids'].tolist()
    longer = restaurants['attributes.Good for Kids'].fillna(0).tolist()
    new_kids_on_the_block = []
    for index, val in enumerate(longer):
        if val == 0:
            new_kids_on_the_block.append(shorter[index])
        else:
            new_kids_on_the_block.append(val)
    restaurants.drop('attributes.Good For Kids', axis=1, inplace=True)
    restaurants['attributes.Good for Kids'] = new_kids_on_the_block
    restaurants.columns = ['restaurant_attributes.accepts_credit_cards', 'restaurant_attributes.ages_allowed', 'restaurant_attributes.alcohol', 'restaurant_attributes.ambience.casual', 'restaurant_attributes.ambience.classy', 'restaurant_attributes.ambience.divey', 'restaurant_attributes.ambience.hipster', 'restaurant_attributes.ambience.intimate', 'restaurant_attributes.ambience.romantic', 'restaurant_attributes.ambience.touristy', 'restaurant_attributes.ambience.trendy', 'restaurant_attributes.ambience.upscale', 'restaurant_attributes.attire', 'restaurant_attributes.byob', 'restaurant_attributes.byob/corkage', 'restaurant_attributes.by_appointment_only', 'restaurant_attributes.caters', 'restaurant_attributes.coat_check', 'restaurant_attributes.corkage', 'restaurant_attributes.delivery', 'restaurant_attributes.dietary_restrictions.dairy-free', 'restaurant_attributes.dietary_restrictions.gluten-free', 'restaurant_attributes.dietary_restrictions.halal', 'restaurant_attributes.dietary_restrictions.kosher', 'restaurant_attributes.dietary_restrictions.soy-free', 'restaurant_attributes.dietary_restrictions.vegan', 'restaurant_attributes.dietary_restrictions.vegetarian', 'restaurant_attributes.dogs_allowed', 'restaurant_attributes.drive-thr', 'restaurant_attributes.good_for_dancing', 'restaurant_attributes.good_for_groups', 'restaurant_attributes.good_for_breakfast', 'restaurant_attributes.good_for_brunch', 'restaurant_attributes.good_for_dessert', 'restaurant_attributes.good_for_dinner', 'restaurant_attributes.good_for_latenight', 'restaurant_attributes.good_for_lunch', 'restaurant_attributes.good_for_kids', 'restaurant_attributes.happy_hour', 'restaurant_attributes.has_tv', 'restaurant_attributes.music.background_music', 'restaurant_attributes.music.dj', 'restaurant_attributes.music.jukebox', 'restaurant_attributes.music.karaoke', 'restaurant_attributes.music.live', 'restaurant_attributes.music.video', 'restaurant_attributes.noise_level', 'restaurant_attributes.open_24_hours', 'restaurant_attributes.order_at_counter', 'restaurant_attributes.outdoor_seating', 'restaurant_attributes.parking.garage', 'restaurant_attributes.parking.lot', 'restaurant_attributes.parking.street', 'restaurant_attributes.parking.valet', 'restaurant_attributes.parking.validated', 'restaurant_attributes.payment_types.amex', 'restaurant_attributes.payment_types.cash_only', 'restaurant_attributes.payment_types.discover', 'restaurant_attributes.payment_types.mastercard', 'restaurant_attributes.payment_types.visa', 'restaurant_attributes.price_range', 'restaurant_attributes.smoking', 'restaurant_attributes.take-out', 'restaurant_attributes.takes_reservations', 'restaurant_attributes.waiter_service', 'restaurant_attributes.wheelchair_accessible', 'restaurant_attributes.wi-fi', 'restaurant_id', 'restaurant_categories', 'restaurant_city', 'restaurant_full_address', 'restaurant_hours.friday.close', 'restaurant_hours.friday.open', 'restaurant_hours.monday.close', 'restaurant_hours.monday.open', 'restaurant_hours.saturday.close', 'restaurant_hours.saturday.open', 'restaurant_hours.sunday.close', 'restaurant_hours.sunday.open', 'restaurant_hours.thursday.close', 'restaurant_hours.thursday.open', 'restaurant_hours.tuesday.close', 'restaurant_hours.tuesday.open', 'restaurant_hours.wednesday.close', 'restaurant_hours.wednesday.open', 'restaurant_latitude', 'restaurant_longitude', 'restaurant_name', 'restaurant_neighborhoods', 'restaurant_open', 'restaurant_review_count', 'restaurant_stars', 'restaurant_state', 'restaurant_type']
    return restaurants


def get_checkins():
    data = []
    with open("data/yelp_academic_dataset_checkin.json", 'r') as f:
        for line in f:
            data.append(byteify(json.loads(line)))
    # checkins = json_normalize(data)
    # above returns like 200 columns of checkin_info
    checkins = pd.DataFrame(data)
    checkins.columns = ['restaurant_id', 'checkin_info', 'checkin_type']
    return checkins


def get_full_features():
    reviews = get_reviews()
    tips = get_tips()

    # some nan's will exist where reviews columns and tips columns don't match up
    reviews_tips = reviews.append(tips)
    reviews_tips.columns = ['restaurant_id', 'review_date', 'tip_likes', 'review_id', 'review_stars', 'review_text', 'review_type', 'user_id', 'review_votes.cool', 'review_votes.funny', 'review_votes.useful']
    # saving this for tfidf vectorizer training later
    with open('models/reviews_tips_original_text.pkl', 'w') as f:
        pickle.dump(reviews_tips.review_text.tolist(), f)

    users = get_users()
    users_reviews_tips = pd.merge(reviews_tips, users, how='left', on='user_id')

    restaurants = get_restaurants()
    restaurants_users_reviews_tips = pd.merge(users_reviews_tips, restaurants, how='outer', on='restaurant_id')

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

    return full_features


def transform_features(df):
    '''
    transform data into workable versions, get rid of unnecessary features.
    format it so that it can be appended to an hdf5 store.
    no mixed objects, no unicode (in python2)
    '''

    # create number representing days passed between inspection date and review date
    df['time_delta'] = (df.inspection_date - df.review_date).astype('timedelta64[D]')

    # weigh scores according to competition weights and sum
    scores = df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']].astype(np.float64)
    df['transformed_score'] = scores.multiply([1, 3, 5], axis=1).sum(axis=1)

    # remove columns that have values without variance or are unnecessary
    df.drop('restaurant_type', axis=1, inplace=True)
    df.drop('review_type', axis=1, inplace=True)
    df.drop('review_id', axis=1, inplace=True)
    df.drop('user_name', axis=1, inplace=True)
    df.drop('user_type', axis=1, inplace=True)
    df.drop('restaurant_state', axis=1, inplace=True)
    df.drop('checkin_type', axis=1, inplace=True)
    # networkx this later
    df.drop('user_friends', axis=1, inplace=True)

    # expand review_date and inspection_date into parts of year. could probably just get by with month or dayofyear
    df['review_year'] = df['inspection_date'].dt.year
    df['review_month'] = df['review_date'].dt.month
    df['review_day'] = df['review_date'].dt.day
    df['review_dayofweek'] = df['review_date'].dt.dayofweek
    df['review_quarter'] = df['review_date'].dt.quarter
    df['review_dayofyear'] = df['review_date'].dt.dayofyear
    df['inspection_year'] = df['inspection_date'].dt.year
    df['inspection_month'] = df['inspection_date'].dt.month
    df['inspection_day'] = df['inspection_date'].dt.day
    df['inspection_dayofweek'] = df['inspection_date'].dt.dayofweek
    df['inspection_quarter'] = df['inspection_date'].dt.quarter
    df['inspection_dayofyear'] = df['inspection_date'].dt.dayofyear

    # convert user_elite to the most recent year
    df['user_most_recent_elite_year'] = df['user_elite'].apply(lambda x: x[-1] if x else np.nan)
    df.drop('user_elite', axis=1, inplace=True)

    # convert to datetime object
    df['user_yelping_since'] = pd.to_datetime(pd.Series(df['user_yelping_since']))

    # convert to bool type
    print('convert to bool type')
    df['restaurant_attributes.accepts_credit_cards'] = df['restaurant_attributes.accepts_credit_cards'].astype('bool')
    df['restaurant_attributes.byob'] = df['restaurant_attributes.byob'].astype('bool')
    df['restaurant_attributes.by_appointment_only'] = df['restaurant_attributes.by_appointment_only'].astype('bool')
    df['restaurant_attributes.caters'] = df['restaurant_attributes.caters'].astype('bool')
    df['restaurant_attributes.coat_check'] = df['restaurant_attributes.coat_check'].astype('bool')
    df['restaurant_attributes.corkage'] = df['restaurant_attributes.corkage'].astype('bool')
    df['restaurant_attributes.delivery'] = df['restaurant_attributes.delivery'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.dairy-free'] = df['restaurant_attributes.dietary_restrictions.dairy-free'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.gluten-free'] = df['restaurant_attributes.dietary_restrictions.gluten-free'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.halal'] = df['restaurant_attributes.dietary_restrictions.halal'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.kosher'] = df['restaurant_attributes.dietary_restrictions.kosher'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.soy-free'] = df['restaurant_attributes.dietary_restrictions.soy-free'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.vegan'] = df['restaurant_attributes.dietary_restrictions.vegan'].astype('bool')
    df['restaurant_attributes.dietary_restrictions.vegetarian'] = df['restaurant_attributes.dietary_restrictions.vegetarian'].astype('bool')
    df['restaurant_attributes.dogs_allowed'] = df['restaurant_attributes.dogs_allowed'].astype('bool')
    df['restaurant_attributes.drive-thr'] = df['restaurant_attributes.drive-thr'].astype('bool')
    df['restaurant_attributes.good_for_dancing'] = df['restaurant_attributes.good_for_dancing'].astype('bool')
    df['restaurant_attributes.good_for_groups'] = df['restaurant_attributes.good_for_groups'].astype('bool')
    df['restaurant_attributes.good_for_breakfast'] = df['restaurant_attributes.good_for_breakfast'].astype('bool')
    df['restaurant_attributes.good_for_brunch'] = df['restaurant_attributes.good_for_brunch'].astype('bool')
    df['restaurant_attributes.good_for_dessert'] = df['restaurant_attributes.good_for_dessert'].astype('bool')
    df['restaurant_attributes.good_for_dinner'] = df['restaurant_attributes.good_for_dinner'].astype('bool')
    df['restaurant_attributes.good_for_latenight'] = df['restaurant_attributes.good_for_latenight'].astype('bool')
    df['restaurant_attributes.good_for_lunch'] = df['restaurant_attributes.good_for_lunch'].astype('bool')
    df['restaurant_attributes.good_for_kids'] = df['restaurant_attributes.good_for_kids'].astype('bool')
    df['restaurant_attributes.happy_hour'] = df['restaurant_attributes.happy_hour'].astype('bool')
    df['restaurant_attributes.has_tv'] = df['restaurant_attributes.has_tv'].astype('bool')
    df['restaurant_attributes.open_24_hours'] = df['restaurant_attributes.open_24_hours'].astype('bool')
    df['restaurant_attributes.order_at_counter'] = df['restaurant_attributes.order_at_counter'].astype('bool')
    df['restaurant_attributes.outdoor_seating'] = df['restaurant_attributes.outdoor_seating'].astype('bool')
    df['restaurant_attributes.payment_types.amex'] = df['restaurant_attributes.payment_types.amex'].astype('bool')
    df['restaurant_attributes.payment_types.cash_only'] = df['restaurant_attributes.payment_types.cash_only'].astype('bool')
    df['restaurant_attributes.payment_types.discover'] = df['restaurant_attributes.payment_types.discover'].astype('bool')
    df['restaurant_attributes.payment_types.mastercard'] = df['restaurant_attributes.payment_types.mastercard'].astype('bool')
    df['restaurant_attributes.payment_types.visa'] = df['restaurant_attributes.payment_types.visa'].astype('bool')
    df['restaurant_attributes.take-out'] = df['restaurant_attributes.take-out'].astype('bool')
    df['restaurant_attributes.takes_reservations'] = df['restaurant_attributes.takes_reservations'].astype('bool')
    df['restaurant_attributes.waiter_service'] = df['restaurant_attributes.waiter_service'].astype('bool')
    df['restaurant_attributes.wheelchair_accessible'] = df['restaurant_attributes.wheelchair_accessible'].astype('bool')

    # make categorical type
    print('make categorical type')
    df['restaurant_attributes.ages_allowed'] = df['restaurant_attributes.ages_allowed'].astype('category')
    df['restaurant_attributes.alcohol'] = df['restaurant_attributes.alcohol'].astype('category')
    df['restaurant_attributes.attire'] = df['restaurant_attributes.attire'].astype('category')
    df['restaurant_attributes.byob/corkage'] = df['restaurant_attributes.byob/corkage'].astype('category')
    df['restaurant_attributes.noise_level'] = df['restaurant_attributes.noise_level'].astype('category')
    df['restaurant_attributes.smoking'] = df['restaurant_attributes.smoking'].astype('category')
    df['restaurant_attributes.wi-fi'] = df['restaurant_attributes.wi-fi'].astype('category')
    df['restaurant_city'] = df['restaurant_city'].astype('category')
    df['restaurant_hours.friday.close'] = df['restaurant_hours.friday.close'].astype('category')
    df['restaurant_hours.friday.open'] = df['restaurant_hours.friday.open'].astype('category')
    df['restaurant_hours.monday.close'] = df['restaurant_hours.monday.close'].astype('category')
    df['restaurant_hours.monday.open'] = df['restaurant_hours.monday.open'].astype('category')
    df['restaurant_hours.saturday.close'] = df['restaurant_hours.saturday.close'].astype('category')
    df['restaurant_hours.saturday.open'] = df['restaurant_hours.saturday.open'].astype('category')
    df['restaurant_hours.sunday.close'] = df['restaurant_hours.sunday.close'].astype('category')
    df['restaurant_hours.sunday.open'] = df['restaurant_hours.sunday.open'].astype('category')
    df['restaurant_hours.thursday.close'] = df['restaurant_hours.thursday.close'].astype('category')
    df['restaurant_hours.thursday.open'] = df['restaurant_hours.thursday.open'].astype('category')
    df['restaurant_hours.tuesday.close'] = df['restaurant_hours.tuesday.close'].astype('category')
    df['restaurant_hours.tuesday.open'] = df['restaurant_hours.tuesday.open'].astype('category')
    df['restaurant_hours.wednesday.close'] = df['restaurant_hours.wednesday.close'].astype('category')
    df['restaurant_hours.wednesday.open'] = df['restaurant_hours.wednesday.open'].astype('category')
    df['restaurant_stars'] = df['restaurant_stars'].astype('category')

    # flatten ambience into one column
    print('flatten ambience into one column')
    casual = df[df['restaurant_attributes.ambience.casual'] == True].index
    classy = df[df['restaurant_attributes.ambience.classy'] == True].index
    divey = df[df['restaurant_attributes.ambience.divey'] == True].index
    hipster = df[df['restaurant_attributes.ambience.hipster'] == True].index
    intimate = df[df['restaurant_attributes.ambience.intimate'] == True].index
    romantic = df[df['restaurant_attributes.ambience.romantic'] == True].index
    touristy = df[df['restaurant_attributes.ambience.touristy'] == True].index
    trendy = df[df['restaurant_attributes.ambience.trendy'] == True].index
    upscale = df[df['restaurant_attributes.ambience.upscale'] == True].index
    df.loc[casual, 'restaurant_ambience'] = 'casual'
    df.loc[classy, 'restaurant_ambience'] = 'classy'
    df.loc[divey, 'restaurant_ambience'] = 'divey'
    df.loc[hipster, 'restaurant_ambience'] = 'hipster'
    df.loc[intimate, 'restaurant_ambience'] = 'intimate'
    df.loc[romantic, 'restaurant_ambience'] = 'romantic'
    df.loc[touristy, 'restaurant_ambience'] = 'touristy'
    df.loc[trendy, 'restaurant_ambience'] = 'trendy'
    df.loc[upscale, 'restaurant_ambience'] = 'upscale'
    df['restaurant_ambience'] = df['restaurant_ambience'].astype('category')
    df.drop(['restaurant_attributes.ambience.casual', 'restaurant_attributes.ambience.classy', 'restaurant_attributes.ambience.divey', 'restaurant_attributes.ambience.hipster', 'restaurant_attributes.ambience.intimate', 'restaurant_attributes.ambience.romantic', 'restaurant_attributes.ambience.touristy', 'restaurant_attributes.ambience.trendy', 'restaurant_attributes.ambience.upscale'], axis=1, inplace=True)

    # flatten music into one column
    print('flatten music into one column')
    background_music = df[df['restaurant_attributes.music.background_music'] == True].index
    dj = df[df['restaurant_attributes.music.dj'] == True].index
    jukebox = df[df['restaurant_attributes.music.jukebox'] == True].index
    karaoke = df[df['restaurant_attributes.music.karaoke'] == True].index
    live = df[df['restaurant_attributes.music.live'] == True].index
    video = df[df['restaurant_attributes.music.video'] == True].index
    df.loc[background_music, 'restaurant_music'] = 'background_music'
    df.loc[dj, 'restaurant_music'] = 'dj'
    df.loc[jukebox, 'restaurant_music'] = 'jukebox'
    df.loc[karaoke, 'restaurant_music'] = 'karaoke'
    df.loc[live, 'restaurant_music'] = 'live'
    df.loc[video, 'restaurant_music'] = 'video'
    df['restaurant_music'] = df['restaurant_music'].astype('category')
    df.drop(['restaurant_attributes.music.background_music', 'restaurant_attributes.music.dj', 'restaurant_attributes.music.jukebox', 'restaurant_attributes.music.karaoke', 'restaurant_attributes.music.live', 'restaurant_attributes.music.video'], axis=1, inplace=True)

    # flatten parking into one column
    print('flatten parking into one column')
    garage = df[df['restaurant_attributes.parking.garage'] == True].index
    lot = df[df['restaurant_attributes.parking.lot'] == True].index
    street = df[df['restaurant_attributes.parking.street'] == True].index
    valet = df[df['restaurant_attributes.parking.valet'] == True].index
    validated = df[df['restaurant_attributes.parking.validated'] == True].index
    df.loc[garage, 'restaurant_parking'] = 'garage'
    df.loc[lot, 'restaurant_parking'] = 'lot'
    df.loc[street, 'restaurant_parking'] = 'street'
    df.loc[valet, 'restaurant_parking'] = 'valet'
    df.loc[validated, 'restaurant_parking'] = 'validated'
    df['restaurant_parking'] = df['restaurant_parking'].astype('category')
    df.drop(['restaurant_attributes.parking.garage', 'restaurant_attributes.parking.lot', 'restaurant_attributes.parking.street', 'restaurant_attributes.parking.valet', 'restaurant_attributes.parking.validated'], axis=1, inplace=True)

    # convert address to just the street name and zip code
    print('convert address to just the street name and zip code')
    df['restaurant_street'] = df['restaurant_full_address'].apply(lambda x: re.search('[A-z].*', x).group() if re.search('[A-z].*', x) is not None else np.nan).astype('category')
    df['restaruant_zipcode'] = df['restaurant_full_address'].apply(lambda x: re.search('\d+$', x).group() if re.search('\d+$', x) is not None else np.nan).astype('category')
    df.drop('restaurant_full_address', axis=1, inplace=True)

    # # convert restaurant_categories into separate columns
    # print('convert restaurant_categories into separate columns')
    # category_list = []
    # for categories in df['restaurant_categories']:
    #     category_list.append({'restaurant_category_'+i.lower().replace(' ', '_'): True for i in categories})
    # category_df = df['restaurant_id'].append(pd.DataFrame(category_list, dtype='bool'))
    # # category_df.to_pickle('models/restaurant_categories_df.pkl')
    # df = df.append(category_df)
    # df.drop('restaurant_categories', axis=1, inplace=True)

    # convert first neighborhood listing in list to the only value
    print('convert first neighborhood listing in list to the only value')
    df['restaurant_neighborhood'] = df['restaurant_neighborhoods'].apply(lambda x: x[0] if x else np.nan)
    df.drop('restaurant_neighborhoods', axis=1, inplace=True)

    # sum the check in values
    print('sum the check in values')
    df['checkin_counts'] = df['checkin_info'].apply(lambda x: np.nan if pd.isnull(x) else sum(x.values()))
    df.drop('checkin_info', axis=1, inplace=True)

    # force text to non-unicode
    print('force text to non-unicode')
    # df['restaurant_id'] = df['restaurant_id'].apply(lambda x: x.decode('utf-8'))
    # df['review_text'] = df['review_text'].apply(lambda x: x.decode('utf-8'))
    df['review_text'] = df['review_text'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore') if type(x) != str else x)
    # df['user_id'] = df['user_id'].apply(lambda x: x.decode('utf-8'))
    # df['restaurant_name'] = df['restaurant_name'].apply(lambda x: x.decode('utf-8'))
    # df['restaurant_neighborhood'] = df['restaurant_neighborhood'].apply(lambda x: x.decode('utf-8'))

    df = df.convert_objects()

    return df


def make_feature_response(feature_df, response_df):
    # convert dates to datetime object
    response_df.inspection_date = pd.to_datetime(pd.Series(response_df.inspection_date))
    feature_df.review_date = pd.to_datetime(pd.Series(feature_df.review_date))
    # combine features and response
    features_response = pd.merge(feature_df, response_df, on='restaurant_id')

    # remove reviews and tips that occur after an inspection - canceled because then end up removing restaurants that we are trying to predict for. future reviews still have predictive power and frames with no reviews but other information still have predictive power
    # no_future = features_response[features_response.review_date < features_response.inspection_date]

    return features_response


def clean_cols_for_hdf(data):
    types = data.apply(lambda x: pd.lib.infer_dtype(x.values))
    for col in types[types=='mixed'].index:
        data[col] = data[col].astype(str)
    # data[<your appropriate columns here>].fillna(0,inplace=True)
    return data


def make_train_test():
    full_features = get_full_features()
    training_response = pd.read_csv("data/train_labels.csv", index_col=None)
    training_response.columns = ['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=None)
    submission.columns = ['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    # combine features and response
    training_df = make_feature_response(full_features, training_response)
    test_df = make_feature_response(full_features, submission)

    # transform dataframes
    print('transforming training set')
    transformed_training_df = transform_features(training_df)
    print('transforming test set')
    transformed_test_df = transform_features(test_df)

    # convert restaurant_id's into numbers representing the restaurant then applying the same
    # categories to the submission dataframe
    restaurant_categories = pd.Categorical.from_array(transformed_training_df.restaurant_id)
    transformed_training_df['restaurant_id_number'] = restaurant_categories.codes
    transformed_test_df['restaurant_id_number'] = restaurant_categories.categories.get_indexer(transformed_test_df.restaurant_id)
    print('finished transformations')

    # save dataframes
    transformed_training_df.to_pickle('models/training_df.pkl')
    transformed_test_df.to_pickle('models/test_df.pkl')
    print('both dataframes pickled')

    # save column/feature names since they have grown out of hand
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(transformed_training_df.columns.tolist()))

    store = pd.HDFStore('models/df_store.h5')
    store.append('training_df', transformed_training_df, data_columns=True, dropna=False)
    print('training_df in hdfstore')
    store.append('test_df', transformed_test_df, data_columns=True, dropna=False)
    store.close()

    return transformed_training_df, transformed_test_df


def load_df(key, columns=None):
    store = pd.HDFStore('models/df_store.h5')
    if not columns:
        df = store.select(key)
    else:
        df = store.select(key, "columns=columns")
    store.close()
    return df


def load_dataframes_selects(column_list):
    column_list.extend(['inspection_id', 'inspection_date', 'restaurant_id', 'time_delta', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3', 'transformed_score'])
    train_df = load_df('training_df', column_list)
    test_df = load_df('test_df', column_list)
    return train_df, test_df


def load_dataframes():
    # test refers to what is being submitted to the competition
    # running cross_val_score on train only
    train_df = load_df('training_df')
    test_df = load_df('test_df')
    return train_df, test_df


def main():
    t0 = time()
    # reviews = get_reviews()
    # train_labels, train_targets = get_response()
    # submission = get_submission()
    # train_and_save(reviews, train_labels, submission)

    make_train_test()
    t1 = time()
    print("{} seconds elapsed.".format(t1 - t0))
    sendMessage.doneTextSend(t0, t1, 'data_grab')


if __name__ == '__main__':
    main()
