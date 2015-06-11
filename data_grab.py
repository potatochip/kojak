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
# from pymongo.cursor import CursorType
from progressbar import ProgressBar
# from pymongo import MongoClient

# client = MongoClient()
# db = client.hygiene


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
    users.columns = ['user_average_stars', 'user_compliments_cool', 'user_compliments_cute', 'user_compliments_funny', 'user_compliments_hot', 'user_compliments_list', 'user_compliments_more', 'user_compliments_note', 'user_compliments_photos', 'user_compliments_plain', 'user_compliments_profile', 'user_compliments_writer', 'user_elite', 'user_fans', 'user_friends', 'user_name', 'user_review_count', 'user_type', 'user_id', 'user_votes_cool', 'user_votes_funny', 'user_votes_useful', 'user_yelping_since']

    # save user info for networkX
    users.to_pickle('models/user_info.pkl')

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
    restaurants.columns = ['restaurant_attributes_accepts_credit_cards', 'restaurant_attributes_ages_allowed', 'restaurant_attributes_alcohol', 'restaurant_attributes_ambience_casual', 'restaurant_attributes_ambience_classy', 'restaurant_attributes_ambience_divey', 'restaurant_attributes_ambience_hipster', 'restaurant_attributes_ambience_intimate', 'restaurant_attributes_ambience_romantic', 'restaurant_attributes_ambience_touristy', 'restaurant_attributes_ambience_trendy', 'restaurant_attributes_ambience_upscale', 'restaurant_attributes_attire', 'restaurant_attributes_byob', 'restaurant_attributes_byob/corkage', 'restaurant_attributes_by_appointment_only', 'restaurant_attributes_caters', 'restaurant_attributes_coat_check', 'restaurant_attributes_corkage', 'restaurant_attributes_delivery', 'restaurant_attributes_dietary_restrictions_dairy-free', 'restaurant_attributes_dietary_restrictions_gluten-free', 'restaurant_attributes_dietary_restrictions_halal', 'restaurant_attributes_dietary_restrictions_kosher', 'restaurant_attributes_dietary_restrictions_soy-free', 'restaurant_attributes_dietary_restrictions_vegan', 'restaurant_attributes_dietary_restrictions_vegetarian', 'restaurant_attributes_dogs_allowed', 'restaurant_attributes_drive-thr', 'restaurant_attributes_good_for_dancing', 'restaurant_attributes_good_for_groups', 'restaurant_attributes_good_for_breakfast', 'restaurant_attributes_good_for_brunch', 'restaurant_attributes_good_for_dessert', 'restaurant_attributes_good_for_dinner', 'restaurant_attributes_good_for_latenight', 'restaurant_attributes_good_for_lunch', 'restaurant_attributes_good_for_kids', 'restaurant_attributes_happy_hour', 'restaurant_attributes_has_tv', 'restaurant_attributes_music_background_music', 'restaurant_attributes_music_dj', 'restaurant_attributes_music_jukebox', 'restaurant_attributes_music_karaoke', 'restaurant_attributes_music_live', 'restaurant_attributes_music_video', 'restaurant_attributes_noise_level', 'restaurant_attributes_open_24_hours', 'restaurant_attributes_order_at_counter', 'restaurant_attributes_outdoor_seating', 'restaurant_attributes_parking_garage', 'restaurant_attributes_parking_lot', 'restaurant_attributes_parking_street', 'restaurant_attributes_parking_valet', 'restaurant_attributes_parking_validated', 'restaurant_attributes_payment_types_amex', 'restaurant_attributes_payment_types_cash_only', 'restaurant_attributes_payment_types_discover', 'restaurant_attributes_payment_types_mastercard', 'restaurant_attributes_payment_types_visa', 'restaurant_attributes_price_range', 'restaurant_attributes_smoking', 'restaurant_attributes_take-out', 'restaurant_attributes_takes_reservations', 'restaurant_attributes_waiter_service', 'restaurant_attributes_wheelchair_accessible', 'restaurant_attributes_wi-fi', 'restaurant_id', 'restaurant_categories', 'restaurant_city', 'restaurant_full_address', 'restaurant_hours_friday_close', 'restaurant_hours_friday_open', 'restaurant_hours_monday_close', 'restaurant_hours_monday_open', 'restaurant_hours_saturday_close', 'restaurant_hours_saturday_open', 'restaurant_hours_sunday_close', 'restaurant_hours_sunday_open', 'restaurant_hours_thursday_close', 'restaurant_hours_thursday_open', 'restaurant_hours_tuesday_close', 'restaurant_hours_tuesday_open', 'restaurant_hours_wednesday_close', 'restaurant_hours_wednesday_open', 'restaurant_latitude', 'restaurant_longitude', 'restaurant_name', 'restaurant_neighborhoods', 'restaurant_open', 'restaurant_review_count', 'restaurant_stars', 'restaurant_state', 'restaurant_type']
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
    reviews_tips.columns = ['restaurant_id', 'review_date', 'tip_likes', 'review_id', 'review_stars', 'review_text', 'review_type', 'user_id', 'review_votes_cool', 'review_votes_funny', 'review_votes_useful']

    # # saving this for tfidf vectorizer training later
    # with open('models/reviews_tips_original_text.pkl', 'w') as f:
    #     pickle.dump(reviews_tips.review_text.tolist(), f)

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

    # remove columns that have values without variance or are unnecessary
    df.drop('restaurant_type', axis=1, inplace=True)
    df.drop('review_type', axis=1, inplace=True)
    df.drop('review_id', axis=1, inplace=True)
    # df.drop('user_name', axis=1, inplace=True)
    df.drop('user_type', axis=1, inplace=True)
    df.drop('restaurant_state', axis=1, inplace=True)
    df.drop('checkin_type', axis=1, inplace=True)
    df.drop('user_friends', axis=1, inplace=True)

    # expand review_date and inspection_date into parts of year. could probably just get by with month or dayofyear
    df.review_date = pd.to_datetime(pd.Series(df.review_date))
    df['review_year'] = df['review_date'].dt.year
    df['review_month'] = df['review_date'].dt.month
    df['review_day'] = df['review_date'].dt.day
    df['review_dayofweek'] = df['review_date'].dt.dayofweek
    df['review_quarter'] = df['review_date'].dt.quarter
    df['review_dayofyear'] = df['review_date'].dt.dayofyear

    # convert user_elite to the most recent year
    df['user_most_recent_elite_year'] = df['user_elite'].apply(lambda x: x[-1] if x else np.nan)
    df.drop('user_elite', axis=1, inplace=True)

    # convert to datetime object
    df['user_yelping_since'] = pd.to_datetime(pd.Series(df['user_yelping_since']))

    # convert to bool type
    print('convert to bool type')
    df['restaurant_attributes_accepts_credit_cards'] = df['restaurant_attributes_accepts_credit_cards'].astype('bool')
    df['restaurant_attributes_byob'] = df['restaurant_attributes_byob'].astype('bool')
    df['restaurant_attributes_by_appointment_only'] = df['restaurant_attributes_by_appointment_only'].astype('bool')
    df['restaurant_attributes_caters'] = df['restaurant_attributes_caters'].astype('bool')
    df['restaurant_attributes_coat_check'] = df['restaurant_attributes_coat_check'].astype('bool')
    df['restaurant_attributes_corkage'] = df['restaurant_attributes_corkage'].astype('bool')
    df['restaurant_attributes_delivery'] = df['restaurant_attributes_delivery'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_dairy-free'] = df['restaurant_attributes_dietary_restrictions_dairy-free'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_gluten-free'] = df['restaurant_attributes_dietary_restrictions_gluten-free'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_halal'] = df['restaurant_attributes_dietary_restrictions_halal'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_kosher'] = df['restaurant_attributes_dietary_restrictions_kosher'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_soy-free'] = df['restaurant_attributes_dietary_restrictions_soy-free'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_vegan'] = df['restaurant_attributes_dietary_restrictions_vegan'].astype('bool')
    df['restaurant_attributes_dietary_restrictions_vegetarian'] = df['restaurant_attributes_dietary_restrictions_vegetarian'].astype('bool')
    df['restaurant_attributes_dogs_allowed'] = df['restaurant_attributes_dogs_allowed'].astype('bool')
    df['restaurant_attributes_drive-thr'] = df['restaurant_attributes_drive-thr'].astype('bool')
    df['restaurant_attributes_good_for_dancing'] = df['restaurant_attributes_good_for_dancing'].astype('bool')
    df['restaurant_attributes_good_for_groups'] = df['restaurant_attributes_good_for_groups'].astype('bool')
    df['restaurant_attributes_good_for_breakfast'] = df['restaurant_attributes_good_for_breakfast'].astype('bool')
    df['restaurant_attributes_good_for_brunch'] = df['restaurant_attributes_good_for_brunch'].astype('bool')
    df['restaurant_attributes_good_for_dessert'] = df['restaurant_attributes_good_for_dessert'].astype('bool')
    df['restaurant_attributes_good_for_dinner'] = df['restaurant_attributes_good_for_dinner'].astype('bool')
    df['restaurant_attributes_good_for_latenight'] = df['restaurant_attributes_good_for_latenight'].astype('bool')
    df['restaurant_attributes_good_for_lunch'] = df['restaurant_attributes_good_for_lunch'].astype('bool')
    df['restaurant_attributes_good_for_kids'] = df['restaurant_attributes_good_for_kids'].astype('bool')
    df['restaurant_attributes_happy_hour'] = df['restaurant_attributes_happy_hour'].astype('bool')
    df['restaurant_attributes_has_tv'] = df['restaurant_attributes_has_tv'].astype('bool')
    df['restaurant_attributes_open_24_hours'] = df['restaurant_attributes_open_24_hours'].astype('bool')
    df['restaurant_attributes_order_at_counter'] = df['restaurant_attributes_order_at_counter'].astype('bool')
    df['restaurant_attributes_outdoor_seating'] = df['restaurant_attributes_outdoor_seating'].astype('bool')
    df['restaurant_attributes_payment_types_amex'] = df['restaurant_attributes_payment_types_amex'].astype('bool')
    df['restaurant_attributes_payment_types_cash_only'] = df['restaurant_attributes_payment_types_cash_only'].astype('bool')
    df['restaurant_attributes_payment_types_discover'] = df['restaurant_attributes_payment_types_discover'].astype('bool')
    df['restaurant_attributes_payment_types_mastercard'] = df['restaurant_attributes_payment_types_mastercard'].astype('bool')
    df['restaurant_attributes_payment_types_visa'] = df['restaurant_attributes_payment_types_visa'].astype('bool')
    df['restaurant_attributes_take-out'] = df['restaurant_attributes_take-out'].astype('bool')
    df['restaurant_attributes_takes_reservations'] = df['restaurant_attributes_takes_reservations'].astype('bool')
    df['restaurant_attributes_waiter_service'] = df['restaurant_attributes_waiter_service'].astype('bool')
    df['restaurant_attributes_wheelchair_accessible'] = df['restaurant_attributes_wheelchair_accessible'].astype('bool')

    # flatten ambience into one column
    print('flatten ambience into one column')
    casual = df[df['restaurant_attributes_ambience_casual'] == True].index
    classy = df[df['restaurant_attributes_ambience_classy'] == True].index
    divey = df[df['restaurant_attributes_ambience_divey'] == True].index
    hipster = df[df['restaurant_attributes_ambience_hipster'] == True].index
    intimate = df[df['restaurant_attributes_ambience_intimate'] == True].index
    romantic = df[df['restaurant_attributes_ambience_romantic'] == True].index
    touristy = df[df['restaurant_attributes_ambience_touristy'] == True].index
    trendy = df[df['restaurant_attributes_ambience_trendy'] == True].index
    upscale = df[df['restaurant_attributes_ambience_upscale'] == True].index
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
    df.drop(['restaurant_attributes_ambience_casual', 'restaurant_attributes_ambience_classy', 'restaurant_attributes_ambience_divey', 'restaurant_attributes_ambience_hipster', 'restaurant_attributes_ambience_intimate', 'restaurant_attributes_ambience_romantic', 'restaurant_attributes_ambience_touristy', 'restaurant_attributes_ambience_trendy', 'restaurant_attributes_ambience_upscale'], axis=1, inplace=True)

    # flatten music into one column
    print('flatten music into one column')
    background_music = df[df['restaurant_attributes_music_background_music'] == True].index
    dj = df[df['restaurant_attributes_music_dj'] == True].index
    jukebox = df[df['restaurant_attributes_music_jukebox'] == True].index
    karaoke = df[df['restaurant_attributes_music_karaoke'] == True].index
    live = df[df['restaurant_attributes_music_live'] == True].index
    video = df[df['restaurant_attributes_music_video'] == True].index
    df.loc[background_music, 'restaurant_music'] = 'background_music'
    df.loc[dj, 'restaurant_music'] = 'dj'
    df.loc[jukebox, 'restaurant_music'] = 'jukebox'
    df.loc[karaoke, 'restaurant_music'] = 'karaoke'
    df.loc[live, 'restaurant_music'] = 'live'
    df.loc[video, 'restaurant_music'] = 'video'
    df['restaurant_music'] = df['restaurant_music'].astype('category')
    df.drop(['restaurant_attributes_music_background_music', 'restaurant_attributes_music_dj', 'restaurant_attributes_music_jukebox', 'restaurant_attributes_music_karaoke', 'restaurant_attributes_music_live', 'restaurant_attributes_music_video'], axis=1, inplace=True)

    # flatten parking into one column
    print('flatten parking into one column')
    garage = df[df['restaurant_attributes_parking_garage'] == True].index
    lot = df[df['restaurant_attributes_parking_lot'] == True].index
    street = df[df['restaurant_attributes_parking_street'] == True].index
    valet = df[df['restaurant_attributes_parking_valet'] == True].index
    validated = df[df['restaurant_attributes_parking_validated'] == True].index
    df.loc[garage, 'restaurant_parking'] = 'garage'
    df.loc[lot, 'restaurant_parking'] = 'lot'
    df.loc[street, 'restaurant_parking'] = 'street'
    df.loc[valet, 'restaurant_parking'] = 'valet'
    df.loc[validated, 'restaurant_parking'] = 'validated'
    df['restaurant_parking'] = df['restaurant_parking'].astype('category')
    df.drop(['restaurant_attributes_parking_garage', 'restaurant_attributes_parking_lot', 'restaurant_attributes_parking_street', 'restaurant_attributes_parking_valet', 'restaurant_attributes_parking_validated'], axis=1, inplace=True)

    # convert address to just the street name and zip code
    print('convert address to just the street name and zip code')
    df['restaurant_street'] = df['restaurant_full_address'].apply(lambda x: re.search('[A-z].*', x).group() if re.search('[A-z].*', x) is not None else np.nan).astype('category')
    df['restaruant_zipcode'] = df['restaurant_full_address'].apply(lambda x: re.search('\d+$', x).group() if re.search('\d+$', x) is not None else np.nan).astype('category')
    # df.drop('restaurant_full_address', axis=1, inplace=True)

    # sum the checkin values
    print('sum the check in values')
    df['checkin_counts'] = df['checkin_info'].apply(lambda x: np.nan if pd.isnull(x) else sum(x.values()))
    df.drop('checkin_info', axis=1, inplace=True)

    # force text to non-unicode
    print('force text to non-unicode')
    df['review_text'] = df['review_text'].apply(lambda x: unicodedata.normalize('NFKD', x) if type(x) != str else x)

    # make categorical type
    print('make categorical type')
    df['restaurant_attributes_ages_allowed'] = df['restaurant_attributes_ages_allowed'].astype('category')
    df['restaurant_attributes_alcohol'] = df['restaurant_attributes_alcohol'].astype('category')
    df['restaurant_attributes_attire'] = df['restaurant_attributes_attire'].astype('category')
    df['restaurant_attributes_byob/corkage'] = df['restaurant_attributes_byob/corkage'].astype('category')
    df['restaurant_attributes_noise_level'] = df['restaurant_attributes_noise_level'].astype('category')
    df['restaurant_attributes_smoking'] = df['restaurant_attributes_smoking'].astype('category')
    df['restaurant_attributes_wi-fi'] = df['restaurant_attributes_wi-fi'].astype('category')
    df['restaurant_city'] = df['restaurant_city'].astype('category')
    df['restaurant_hours_friday_close'] = df['restaurant_hours_friday_close'].astype('category')
    df['restaurant_hours_friday_open'] = df['restaurant_hours_friday_open'].astype('category')
    df['restaurant_hours_monday_close'] = df['restaurant_hours_monday_close'].astype('category')
    df['restaurant_hours_monday_open'] = df['restaurant_hours_monday_open'].astype('category')
    df['restaurant_hours_saturday_close'] = df['restaurant_hours_saturday_close'].astype('category')
    df['restaurant_hours_saturday_open'] = df['restaurant_hours_saturday_open'].astype('category')
    df['restaurant_hours_sunday_close'] = df['restaurant_hours_sunday_close'].astype('category')
    df['restaurant_hours_sunday_open'] = df['restaurant_hours_sunday_open'].astype('category')
    df['restaurant_hours_thursday_close'] = df['restaurant_hours_thursday_close'].astype('category')
    df['restaurant_hours_thursday_open'] = df['restaurant_hours_thursday_open'].astype('category')
    df['restaurant_hours_tuesday_close'] = df['restaurant_hours_tuesday_close'].astype('category')
    df['restaurant_hours_tuesday_open'] = df['restaurant_hours_tuesday_open'].astype('category')
    df['restaurant_hours_wednesday_close'] = df['restaurant_hours_wednesday_close'].astype('category')
    df['restaurant_hours_wednesday_open'] = df['restaurant_hours_wednesday_open'].astype('category')
    df['restaurant_name'] = df['restaurant_name'].astype('category')
    df['user_id'] = df['user_id'].astype('category')
    df['user_name'] = df['user_name'].astype('category')

    # make review_text categorical to make it easier to work with
    df['review_text'] = df['review_text'].astype('category')

    # expand neighborhoods out
    print('expand neighborhoods out')
    neigh_df = pd.DataFrame(df['restaurant_neighborhoods'].tolist(), columns=['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3'])
    cats = neigh_df.restaurant_neighborhood_1.astype('category').cat.categories.tolist() + neigh_df.restaurant_neighborhood_2.astype('category').cat.categories.tolist() + neigh_df.restaurant_neighborhood_3.astype('category').cat.categories.tolist()
    neigh_df['restaurant_neighborhood_1'] = neigh_df['restaurant_neighborhood_1'].astype('category', categories=set(cats))
    neigh_df['restaurant_neighborhood_2'] = neigh_df['restaurant_neighborhood_2'].astype('category', categories=set(cats))
    neigh_df['restaurant_neighborhood_3'] = neigh_df['restaurant_neighborhood_3'].astype('category', categories=set(cats))
    df = pd.concat([df, neigh_df], axis=1, join_axes=[df.index])
    df.drop('restaurant_neighborhoods', axis=1, inplace=True)

    # expand restaurant categories out
    print('expand restaurant categories out')
    categ_df = pd.DataFrame(df['restaurant_categories'].tolist(), columns=['restaurant_category_1', 'restaurant_category_2', 'restaurant_category_3', 'restaurant_category_4', 'restaurant_category_5', 'restaurant_category_6', 'restaurant_category_7'])
    cats = categ_df.restaurant_category_1.astype('category').cat.categories.tolist() + categ_df.restaurant_category_2.astype('category').cat.categories.tolist() + categ_df.restaurant_category_3.astype('category').cat.categories.tolist() + categ_df.restaurant_category_4.astype('category').cat.categories.tolist() + categ_df.restaurant_category_5.astype('category').cat.categories.tolist() + categ_df.restaurant_category_6.astype('category').cat.categories.tolist() + categ_df.restaurant_category_7.astype('category').cat.categories.tolist()
    categ_df['restaurant_category_1'] = categ_df['restaurant_category_1'].astype('category', categories=set(cats))
    categ_df['restaurant_category_2'] = categ_df['restaurant_category_2'].astype('category', categories=set(cats))
    categ_df['restaurant_category_3'] = categ_df['restaurant_category_3'].astype('category', categories=set(cats))
    categ_df['restaurant_category_4'] = categ_df['restaurant_category_4'].astype('category', categories=set(cats))
    categ_df['restaurant_category_5'] = categ_df['restaurant_category_5'].astype('category', categories=set(cats))
    categ_df['restaurant_category_6'] = categ_df['restaurant_category_6'].astype('category', categories=set(cats))
    categ_df['restaurant_category_7'] = categ_df['restaurant_category_7'].astype('category', categories=set(cats))
    df = pd.concat([df, categ_df], axis=1, join_axes=[df.index])
    df.drop('restaurant_categories', axis=1, inplace=True)

    return df


def make_feature_response(feature_df, response_df):
    # convert dates to datetime object
    response_df.inspection_date = pd.to_datetime(pd.Series(response_df.inspection_date))

    # combine features and response
    features_response = pd.merge(feature_df, response_df, on='restaurant_id', how='right')

    # remove reviews and tips that occur after an inspection - canceled because then end up removing restaurants that we are trying to predict for. future reviews still have predictive power and frames with no reviews but other information still have predictive power
    # no_future = features_response[features_response.review_date < features_response.inspection_date]

    return features_response


def make_train_test():
    full_features = get_full_features()

    # transform features
    print('transforming features')
    transformed_features = transform_features(full_features)
    print('finished transformations')

    training_response = pd.read_csv("data/train_labels.csv", index_col=None)
    training_response.columns = ['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=None)
    submission.columns = ['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    # combine features and response
    training_df = make_feature_response(transformed_features, training_response)
    test_df = make_feature_response(transformed_features, submission)

    # convert restaurant_id's into numbers representing the restaurant with same categories
    cats = training_df.restaurant_id.astype('category').cat.categories.tolist() + test_df.restaurant_id.astype('category').cat.categories.tolist()
    training_df['restaurant_id'] = training_df['restaurant_id'].astype('category', categories=set(cats))
    test_df['restaurant_id'] = test_df['restaurant_id'].astype('category', categories=set(cats))
    # restaurant_categories = pd.Categorical.from_array(train_df.restaurant_id)
    # train_df['restaurant_id_number'] = restaurant_categories.codes
    # test_df['restaurant_id_number'] = restaurant_categories.categories.get_indexer(test_df.restaurant_id)

    # create number representing days passed between inspection date and review date
    training_df['time_delta'] = (training_df.inspection_date - training_df.review_date).astype('timedelta64[D]')
    test_df['time_delta'] = (test_df.inspection_date - test_df.review_date).astype('timedelta64[D]')

    # transform inspection date
    training_df['inspection_year'] = training_df['inspection_date'].dt.year
    training_df['inspection_month'] = training_df['inspection_date'].dt.month
    training_df['inspection_day'] = training_df['inspection_date'].dt.day
    training_df['inspection_dayofweek'] = training_df['inspection_date'].dt.dayofweek
    training_df['inspection_quarter'] = training_df['inspection_date'].dt.quarter
    training_df['inspection_dayofyear'] = training_df['inspection_date'].dt.dayofyear

    test_df['inspection_year'] = test_df['inspection_date'].dt.year
    test_df['inspection_month'] = test_df['inspection_date'].dt.month
    test_df['inspection_day'] = test_df['inspection_date'].dt.day
    test_df['inspection_dayofweek'] = test_df['inspection_date'].dt.dayofweek
    test_df['inspection_quarter'] = test_df['inspection_date'].dt.quarter
    test_df['inspection_dayofyear'] = test_df['inspection_date'].dt.dayofyear

    # # save dataframes
    training_df.to_pickle('models/training_df.pkl')
    test_df.to_pickle('models/test_df.pkl')
    print('both dataframes pickled')

    # save column/feature names since they have grown out of hand
    choices_choices = [str(j)+' - '+str(k) for j, k in zip(training_df.dtypes.index, training_df.dtypes)]
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(choices_choices))

    # store = pd.HDFStore('models/df_store.h5')
    # store.append('training_df', transformed_training_df, data_columns=True, dropna=False)
    # print('training_df in hdfstore')
    # store.append('test_df', transformed_test_df, data_columns=True, dropna=False)
    # store.close()
    # dear_mongo(transformed_training_df, 'train')
    # dear_mongo(transformed_test_df, 'test')


# def dear_mongo(df, collection):
#     print('working on {}'.format(collection))
#     pbar = ProgressBar(maxval=df.shape[0]).start()
#     for index, row in enumerate(df.iterrows()):
#         record = json.loads(row[1].to_json())
#         db[collection].save(record)
#         pbar.update(index)
#     pbar.finish()


# def fetch_mongo(collection, features):
#     print("Getting documents")
#     features.extend(['inspection_id', 'inspection_date', 'restaurant_id', 'time_delta', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3', 'transformed_score'])
#     selection = {i: 1 for i in features}.update({"_id": 0})
#     cursor = db[collection].find({}, selection, no_cursor_timeout=True, cursor_type=CursorType.EXHAUST)
#     pbar = ProgressBar(maxval=cursor.count()).start()
#     df = pd.DataFrame()
#     for index, row in enumerate(cursor):
#         df = df.append(row, ignore_index=True)
#         pbar.update(index)
#     pbar.finish()
#     cursor.close()
#     return df


def get_selects(frame, features=None):
    if frame == 'train':
        df = pd.read_pickle('models/training_df.pkl')
        if features:
            features = features[:]
            features.extend(['score_lvl_1', 'score_lvl_2', 'score_lvl_3'])
            return df[features]
        else:
            return df
    elif frame == 'test':
        df = pd.read_pickle('models/test_df.pkl')
        if features:
            features = features[:]
            features.extend(['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3'])
            return df[features]
        else:
            return df


def test():
    df = pd.read_pickle('models/test_df.pkl')
    store = pd.HDFStore('models/df_store.h5')
    store.append('test_df', df, dropna=False, data_columns=['restaurant_id', 'restaurant_full_address', 'review_text', 'user_id'])
    store.close()


def load_dataframes(features=None):
    train_df = get_selects('train', features)
    test_df = get_selects('test', features)
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
