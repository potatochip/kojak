from time import time

import numpy as np
import pandas as pd
import cPickle as pickle
import unicodedata
from pandas.io.json import json_normalize
import sendMessage
import json
import re
from progressbar import ProgressBar


def byteify(input):
    if isinstance(input, dict):
        return {byteify(key): byteify(value) for key, value in input.iteritems()}
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
    # renaming tip.likes to what would be equivalent in reviews
    tips = tips.rename(columns={'likes': 'votes.useful'})
    return tips


def get_users():
    data = []
    with open("data/yelp_academic_dataset_user.json", 'r') as f:
        for line in f:
            data.append(byteify(json.loads(line)))
    users = json_normalize(data)
    users.columns = ['user_average_stars', 'user_compliments_cool', 'user_compliments_cute', 'user_compliments_funny', 'user_compliments_hot', 'user_compliments_list', 'user_compliments_more', 'user_compliments_note', 'user_compliments_photos', 'user_compliments_plain', 'user_compliments_profile', 'user_compliments_writer', 'user_elite', 'user_fans', 'user_friends', 'user_name', 'user_review_count', 'user_type', 'user_id', 'user_votes_cool', 'user_votes_funny', 'user_votes_useful', 'user_yelping_since']

    # save user info for networkX
    users.to_pickle('pickle_jar/user_info.pkl')

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
    restaurants.columns = ['restaurant_attributes_accepts_credit_cards', 'restaurant_attributes_ages_allowed', 'restaurant_attributes_alcohol', 'restaurant_attributes_ambience_casual', 'restaurant_attributes_ambience_classy', 'restaurant_attributes_ambience_divey', 'restaurant_attributes_ambience_hipster', 'restaurant_attributes_ambience_intimate', 'restaurant_attributes_ambience_romantic', 'restaurant_attributes_ambience_touristy', 'restaurant_attributes_ambience_trendy', 'restaurant_attributes_ambience_upscale', 'restaurant_attributes_attire', 'restaurant_attributes_byob', 'restaurant_attributes_byob_corkage', 'restaurant_attributes_by_appointment_only', 'restaurant_attributes_caters', 'restaurant_attributes_coat_check', 'restaurant_attributes_corkage', 'restaurant_attributes_delivery', 'restaurant_attributes_dietary_restrictions_dairy_free', 'restaurant_attributes_dietary_restrictions_gluten_free', 'restaurant_attributes_dietary_restrictions_halal', 'restaurant_attributes_dietary_restrictions_kosher', 'restaurant_attributes_dietary_restrictions_soy_free', 'restaurant_attributes_dietary_restrictions_vegan', 'restaurant_attributes_dietary_restrictions_vegetarian', 'restaurant_attributes_dogs_allowed', 'restaurant_attributes_drive_thr', 'restaurant_attributes_good_for_dancing', 'restaurant_attributes_good_for_groups', 'restaurant_attributes_good_for_breakfast', 'restaurant_attributes_good_for_brunch', 'restaurant_attributes_good_for_dessert', 'restaurant_attributes_good_for_dinner', 'restaurant_attributes_good_for_latenight', 'restaurant_attributes_good_for_lunch', 'restaurant_attributes_good_for_kids', 'restaurant_attributes_happy_hour', 'restaurant_attributes_has_tv', 'restaurant_attributes_music_background_music', 'restaurant_attributes_music_dj', 'restaurant_attributes_music_jukebox', 'restaurant_attributes_music_karaoke', 'restaurant_attributes_music_live', 'restaurant_attributes_music_video', 'restaurant_attributes_noise_level', 'restaurant_attributes_open_24_hours', 'restaurant_attributes_order_at_counter', 'restaurant_attributes_outdoor_seating', 'restaurant_attributes_parking_garage', 'restaurant_attributes_parking_lot', 'restaurant_attributes_parking_street', 'restaurant_attributes_parking_valet', 'restaurant_attributes_parking_validated', 'restaurant_attributes_payment_types_amex', 'restaurant_attributes_payment_types_cash_only', 'restaurant_attributes_payment_types_discover', 'restaurant_attributes_payment_types_mastercard', 'restaurant_attributes_payment_types_visa', 'restaurant_attributes_price_range', 'restaurant_attributes_smoking', 'restaurant_attributes_take_out', 'restaurant_attributes_takes_reservations', 'restaurant_attributes_waiter_service', 'restaurant_attributes_wheelchair_accessible', 'restaurant_attributes_wifi', 'restaurant_id', 'restaurant_categories', 'restaurant_city', 'restaurant_full_address', 'restaurant_hours_friday_close', 'restaurant_hours_friday_open', 'restaurant_hours_monday_close', 'restaurant_hours_monday_open', 'restaurant_hours_saturday_close', 'restaurant_hours_saturday_open', 'restaurant_hours_sunday_close', 'restaurant_hours_sunday_open', 'restaurant_hours_thursday_close', 'restaurant_hours_thursday_open', 'restaurant_hours_tuesday_close', 'restaurant_hours_tuesday_open', 'restaurant_hours_wednesday_close', 'restaurant_hours_wednesday_open', 'restaurant_latitude', 'restaurant_longitude', 'restaurant_name', 'restaurant_neighborhoods', 'restaurant_open', 'restaurant_review_count', 'restaurant_stars', 'restaurant_state', 'restaurant_type']

    # convert opening and closing hours to float representation. will take forever if done after everything is multiplied
    openclose = lambda x: pd.to_datetime(x).hour + pd.to_datetime(x).minute/60.
    restaurants['restaurant_hours_friday_close'] = restaurants['restaurant_hours_friday_close'].apply(openclose)
    restaurants['restaurant_hours_friday_open'] = restaurants['restaurant_hours_friday_open'].apply(openclose)
    restaurants['restaurant_hours_monday_close'] = restaurants['restaurant_hours_monday_close'].apply(openclose)
    restaurants['restaurant_hours_monday_open'] = restaurants['restaurant_hours_monday_open'].apply(openclose)
    restaurants['restaurant_hours_saturday_close'] = restaurants['restaurant_hours_saturday_close'].apply(openclose)
    restaurants['restaurant_hours_saturday_open'] = restaurants['restaurant_hours_saturday_open'].apply(openclose)
    restaurants['restaurant_hours_sunday_close'] = restaurants['restaurant_hours_sunday_close'].apply(openclose)
    restaurants['restaurant_hours_sunday_open'] = restaurants['restaurant_hours_sunday_open'].apply(openclose)
    restaurants['restaurant_hours_thursday_close'] = restaurants['restaurant_hours_thursday_close'].apply(openclose)
    restaurants['restaurant_hours_thursday_open'] = restaurants['restaurant_hours_thursday_open'].apply(openclose)
    restaurants['restaurant_hours_tuesday_close'] = restaurants['restaurant_hours_tuesday_close'].apply(openclose)
    restaurants['restaurant_hours_tuesday_open'] = restaurants['restaurant_hours_tuesday_open'].apply(openclose)
    restaurants['restaurant_hours_wednesday_close'] = restaurants['restaurant_hours_wednesday_close'].apply(openclose)
    restaurants['restaurant_hours_wednesday_open'] = restaurants['restaurant_hours_wednesday_open'].apply(openclose)

    # map to boston inspection ids. yelp has multiple ids referring to the same boston id. condencing multiples into a single row combinging the rows that have the most information
    restaurants = map_ids(restaurants)
    stars = restaurants.groupby('restaurant_id')['restaurant_stars'].median()
    therest = restaurants.drop('restaurant_stars', axis=1).groupby('restaurant_id').max()
    final = pd.concat([stars, therest], axis=1).reset_index()

    return final


def get_checkins():
    data = []
    with open("data/yelp_academic_dataset_checkin.json", 'r') as f:
        for line in f:
            data.append(byteify(json.loads(line)))
    # checkins = json_normalize(data)
    # above returns like 200 columns of checkin_info
    checkins = pd.DataFrame(data)
    checkins.columns = ['restaurant_id', 'checkin_info', 'checkin_type']

    # sum the checkin values
    print('sum the check in values')
    checkins['checkin_counts'] = checkins['checkin_info'].apply(lambda x: np.nan if pd.isnull(x) else sum(x.values()))
    checkins.drop('checkin_info', axis=1, inplace=True)

    # map to boston inspection ids. yelp has multiple ids referring to the same boston id. concening multiples into a single row with the sum count of the number of checkins
    checkins = map_ids(checkins)
    checkins = checkins.groupby('restaurant_id').sum().reset_index()

    return checkins


def map_ids(df):
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

    print("shape before mapping ids: {}".format(df.shape))
    df.restaurant_id = df.restaurant_id.map(map_to_boston_ids)
    print("shape after mapping ids: {}".format(df.shape))
    return df


def get_full_features():
    reviews = get_reviews()
    tips = get_tips()

    reviews_tips = reviews.append(tips)
    reviews_tips.columns = ['restaurant_id', 'review_date', 'review_id', 'review_stars', 'review_text', 'review_type', 'user_id', 'review_votes_cool', 'review_votes_funny', 'review_votes_useful']
    reviews_tips.review_votes_useful.fillna(0, inplace=True)
    reviews_tips.review_votes_cool.fillna(0, inplace=True)
    reviews_tips.review_votes_funny.fillna(0, inplace=True)
    reviews_tips = map_ids(reviews_tips)

    # # saving this for tfidf vectorizer training later
    # with open('pickle_jar/reviews_tips_original_text.pkl', 'w') as f:
    #     pickle.dump(reviews_tips.review_text.tolist(), f)

    users = get_users()
    users_reviews_tips = pd.merge(reviews_tips, users, how='left', on='user_id')

    restaurants = get_restaurants()
    restaurants_users_reviews_tips = pd.merge(users_reviews_tips, restaurants, how='outer', on='restaurant_id')

    # if checkins dont exist for a restaurant dont want to drop the restaurant values
    checkins = get_checkins()
    full_features = pd.merge(restaurants_users_reviews_tips, checkins, how='left', on='restaurant_id')

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
    # df.drop('review_id', axis=1, inplace=True)
    # df.drop('user_name', axis=1, inplace=True)
    df.drop('user_type', axis=1, inplace=True)
    df.drop('restaurant_state', axis=1, inplace=True)
    df.drop('user_friends', axis=1, inplace=True)

    print('converting time-related features')
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

    # convert user_yelping_since and user_most_recent_elite_year to deltas
    df['user_yelping_since_delta'] = (df.review_date - df.user_yelping_since).astype('timedelta64[D]')
    df.drop('user_yelping_since', axis=1, inplace=True)

    df['user_most_recent_elite_year_delta'] = (df.review_date.dt.year - df.user_most_recent_elite_year)
    df['user_ever_elite'] = pd.notnull(df.user_most_recent_elite_year_delta)
    df.drop('user_most_recent_elite_year', axis=1, inplace=True)

    # convert to bool type
    print('convert to bool type')
    df = easy_bools(df, 'restaurant_attributes_accepts_credit_cards')
    df = easy_bools(df, 'restaurant_attributes_byob')
    df = easy_bools(df, 'restaurant_attributes_by_appointment_only')
    df = easy_bools(df, 'restaurant_attributes_caters')
    df = easy_bools(df, 'restaurant_attributes_coat_check')
    df = easy_bools(df, 'restaurant_attributes_corkage')
    df = easy_bools(df, 'restaurant_attributes_delivery')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_dairy_free')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_gluten_free')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_halal')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_kosher')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_soy_free')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_vegan')
    df = easy_bools(df, 'restaurant_attributes_dietary_restrictions_vegetarian')
    df = easy_bools(df, 'restaurant_attributes_dogs_allowed')
    df = easy_bools(df, 'restaurant_attributes_drive_thr')
    df = easy_bools(df, 'restaurant_attributes_good_for_dancing')
    df = easy_bools(df, 'restaurant_attributes_good_for_groups')
    df = easy_bools(df, 'restaurant_attributes_good_for_breakfast')
    df = easy_bools(df, 'restaurant_attributes_good_for_brunch')
    df = easy_bools(df, 'restaurant_attributes_good_for_dessert')
    df = easy_bools(df, 'restaurant_attributes_good_for_dinner')
    df = easy_bools(df, 'restaurant_attributes_good_for_latenight')
    df = easy_bools(df, 'restaurant_attributes_good_for_lunch')
    df = easy_bools(df, 'restaurant_attributes_good_for_kids')
    df = easy_bools(df, 'restaurant_attributes_happy_hour')
    df = easy_bools(df, 'restaurant_attributes_has_tv')
    df = easy_bools(df, 'restaurant_attributes_open_24_hours')
    df = easy_bools(df, 'restaurant_attributes_order_at_counter')
    df = easy_bools(df, 'restaurant_attributes_outdoor_seating')
    df = easy_bools(df, 'restaurant_attributes_payment_types_amex')
    df = easy_bools(df, 'restaurant_attributes_payment_types_cash_only')
    df = easy_bools(df, 'restaurant_attributes_payment_types_discover')
    df = easy_bools(df, 'restaurant_attributes_payment_types_mastercard')
    df = easy_bools(df, 'restaurant_attributes_payment_types_visa')
    df = easy_bools(df, 'restaurant_attributes_take_out')
    df = easy_bools(df, 'restaurant_attributes_takes_reservations')
    df = easy_bools(df, 'restaurant_attributes_waiter_service')
    df = easy_bools(df, 'restaurant_attributes_wheelchair_accessible')

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
    df.drop(['restaurant_attributes_parking_garage', 'restaurant_attributes_parking_lot', 'restaurant_attributes_parking_street', 'restaurant_attributes_parking_valet', 'restaurant_attributes_parking_validated'], axis=1, inplace=True)

    # convert address to just the street name and zip code
    print('convert address to just the street name and zip code')
    df['restaurant_street'] = df['restaurant_full_address'].apply(lambda x: re.search('[A-z].*', x).group() if re.search('[A-z].*', x) is not None else np.nan)
    df['restaurant_zipcode'] = df['restaurant_full_address'].apply(lambda x: re.search('\d+$', x).group() if re.search('\d+$', x) is not None else np.nan)
    # df.drop('restaurant_full_address', axis=1, inplace=True)

    # misc
    df['review_stars'] = df.review_stars.fillna(0).astype('category')
    df.restaurant_attributes_price_range = df.restaurant_attributes_price_range.fillna(df.restaurant_attributes_price_range.median())

    # fix jacked up text
    print('fix jacked up text')
    df['review_text'] = df['review_text'].apply(lambda x: unicodedata.normalize('NFKD', x) if type(x) != str else x)

    return df


def make_feature_response(feature_df, response_df):
    # convert dates to datetime object
    response_df.inspection_date = pd.to_datetime(pd.Series(response_df.inspection_date))

    # combine features and response
    features_response = pd.merge(feature_df, response_df, on='restaurant_id', how='right')

    return features_response


def easy_bools(df, column):
    # converts nans to false
    df[column] = df[column].fillna(False).astype('bool')
    return df


def easy_categories(train_df, test_df, column):
    cats = train_df[column].astype('category').cat.categories.tolist() + test_df[column].astype('category').cat.categories.tolist()
    train_df[column] = train_df[column].astype('category', categories=set(cats))
    test_df[column] = test_df[column].astype('category', categories=set(cats))
    return train_df, test_df


def make_categoricals(train_df, test_df):
    # make categorical type
    print('make categorical types')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_ages_allowed')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_alcohol')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_attire')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_byob')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_byob_corkage')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_noise_level')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_smoking')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_attributes_wifi')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_city')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_name')
    train_df, test_df = easy_categories(train_df, test_df, column='user_id')
    train_df, test_df = easy_categories(train_df, test_df, column='user_name')
    # i commented the below out at some point... why was that. usually need to use df.restaurant_id.convert_objects() if want to work with it again
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_id')

    # make review_text categorical to make it easier to work with
    train_df, test_df = easy_categories(train_df, test_df, column='review_text')

    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_ambience')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_music')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_parking')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_street')
    train_df, test_df = easy_categories(train_df, test_df, column='restaurant_zipcode')

    # expand neighborhoods out
    print('expand neighborhoods out')
    train_temp_df = pd.DataFrame(train_df['restaurant_neighborhoods'].tolist(), columns=['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3'])
    test_temp_df = pd.DataFrame(test_df['restaurant_neighborhoods'].tolist(), columns=['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3'])
    cats = train_temp_df.restaurant_neighborhood_1.astype('category').cat.categories.tolist() + train_temp_df.restaurant_neighborhood_2.astype('category').cat.categories.tolist() + train_temp_df.restaurant_neighborhood_3.astype('category').cat.categories.tolist() + test_temp_df.restaurant_neighborhood_1.astype('category').cat.categories.tolist() + test_temp_df.restaurant_neighborhood_2.astype('category').cat.categories.tolist() + test_temp_df.restaurant_neighborhood_3.astype('category').cat.categories.tolist()
    train_temp_df['restaurant_neighborhood_1'] = train_temp_df['restaurant_neighborhood_1'].astype('category', categories=set(cats))
    train_temp_df['restaurant_neighborhood_2'] = train_temp_df['restaurant_neighborhood_2'].astype('category', categories=set(cats))
    train_temp_df['restaurant_neighborhood_3'] = train_temp_df['restaurant_neighborhood_3'].astype('category', categories=set(cats))
    train_df = pd.concat([train_df, train_temp_df], axis=1, join_axes=[train_df.index])
    # train_df.drop('restaurant_neighborhoods', axis=1, inplace=True)
    test_temp_df['restaurant_neighborhood_1'] = test_temp_df['restaurant_neighborhood_1'].astype('category', categories=set(cats))
    test_temp_df['restaurant_neighborhood_2'] = test_temp_df['restaurant_neighborhood_2'].astype('category', categories=set(cats))
    test_temp_df['restaurant_neighborhood_3'] = test_temp_df['restaurant_neighborhood_3'].astype('category', categories=set(cats))
    test_df = pd.concat([test_df, test_temp_df], axis=1, join_axes=[test_df.index])
    # test_df.drop('restaurant_neighborhoods', axis=1, inplace=True)

    # expand restaurant categories out
    print('expand restaurant categories out')
    train_temp_df = pd.DataFrame(train_df['restaurant_categories'].tolist(), columns=['restaurant_category_1', 'restaurant_category_2', 'restaurant_category_3', 'restaurant_category_4', 'restaurant_category_5', 'restaurant_category_6', 'restaurant_category_7'])
    test_temp_df = pd.DataFrame(test_df['restaurant_categories'].tolist(), columns=['restaurant_category_1', 'restaurant_category_2', 'restaurant_category_3', 'restaurant_category_4', 'restaurant_category_5', 'restaurant_category_6', 'restaurant_category_7'])
    cats = train_temp_df.restaurant_category_1.astype('category').cat.categories.tolist() + train_temp_df.restaurant_category_2.astype('category').cat.categories.tolist() + train_temp_df.restaurant_category_3.astype('category').cat.categories.tolist() + train_temp_df.restaurant_category_4.astype('category').cat.categories.tolist() + train_temp_df.restaurant_category_5.astype('category').cat.categories.tolist() + train_temp_df.restaurant_category_6.astype('category').cat.categories.tolist() + train_temp_df.restaurant_category_7.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_1.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_2.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_3.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_4.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_5.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_6.astype('category').cat.categories.tolist() + test_temp_df.restaurant_category_7.astype('category').cat.categories.tolist()
    train_temp_df['restaurant_category_1'] = train_temp_df['restaurant_category_1'].astype('category', categories=set(cats))
    train_temp_df['restaurant_category_2'] = train_temp_df['restaurant_category_2'].astype('category', categories=set(cats))
    train_temp_df['restaurant_category_3'] = train_temp_df['restaurant_category_3'].astype('category', categories=set(cats))
    train_temp_df['restaurant_category_4'] = train_temp_df['restaurant_category_4'].astype('category', categories=set(cats))
    train_temp_df['restaurant_category_5'] = train_temp_df['restaurant_category_5'].astype('category', categories=set(cats))
    train_temp_df['restaurant_category_6'] = train_temp_df['restaurant_category_6'].astype('category', categories=set(cats))
    train_temp_df['restaurant_category_7'] = train_temp_df['restaurant_category_7'].astype('category', categories=set(cats))
    train_df = pd.concat([train_df, train_temp_df], axis=1, join_axes=[train_df.index])
    # train_df.drop('restaurant_categories', axis=1, inplace=True)
    test_temp_df['restaurant_category_1'] = test_temp_df['restaurant_category_1'].astype('category', categories=set(cats))
    test_temp_df['restaurant_category_2'] = test_temp_df['restaurant_category_2'].astype('category', categories=set(cats))
    test_temp_df['restaurant_category_3'] = test_temp_df['restaurant_category_3'].astype('category', categories=set(cats))
    test_temp_df['restaurant_category_4'] = test_temp_df['restaurant_category_4'].astype('category', categories=set(cats))
    test_temp_df['restaurant_category_5'] = test_temp_df['restaurant_category_5'].astype('category', categories=set(cats))
    test_temp_df['restaurant_category_6'] = test_temp_df['restaurant_category_6'].astype('category', categories=set(cats))
    test_temp_df['restaurant_category_7'] = test_temp_df['restaurant_category_7'].astype('category', categories=set(cats))
    test_df = pd.concat([test_df, test_temp_df], axis=1, join_axes=[test_df.index])
    # test_df.drop('restaurant_categories', axis=1, inplace=True)

    return train_df, test_df


def make_flat_version(df):
    '''
    combining all the reviews for each restaurant/inspection into a single text. will have the same number of rows as the original response. this way we can avoid hierarchical models and test whether it makes a difference
    '''
    # groupby restaurant_id and inspection_date
    g = df[['restaurant_id', 'inspection_date', 'review_text', 'review_date']].groupby(['restaurant_id', 'inspection_date'])
    # remove the reviews that occur after the inspection date and combine reviews for the same restaurant/date
    # texts = g.apply(lambda x: ' '.join(x[x.review_date <= x.inspection_date]['review_text']))
    texts = g.review_text.apply(flatten_texts)
    # remove duplicates
    no_dupes = df.drop_duplicates(['restaurant_id', 'inspection_date'])
    no_dupes.set_index(['restaurant_id', 'inspection_date'], inplace=True)
    no_dupes.review_text = texts
    no_dupes.reset_index(inplace=True)
    print("New shape of {}".format(no_dupes.shape))
    return no_dupes

def flatten_texts(texts):
    try:
        return ' '.join(texts)
    except:
        return np.nan


def post_transformations(df):
    '''transformations that need to occur after everything else is finished. usually after combined with response
    '''

    # create number representing days passed between inspection date and review date
    df['review_delta'] = (df.inspection_date - df.review_date).astype('timedelta64[D]')

    # create number representing days passed since last inspection date and current inspection date. first entry for a restaurant is set at 0 delta
    temp_df = df[['restaurant_id', 'inspection_date']]
    temp_df['temp_date'] = temp_df['inspection_date']
    temp_df.restaurant_id = temp_df.restaurant_id.convert_objects()
    g = temp_df.groupby(['restaurant_id', 'inspection_date'])
    # diff doesnt work witout calling first or max or min or whatever first
    delta = g.temp_date.first().diff()
    for i in delta.index.levels[0]:
        delta[i][0] = 0  # won't allow np.nan or pd.NaT directly
    # delta.replace(-1, np.nan, inplace=True)
    # # pd.merge resets all the datatypes so doing this instead. takes FOREVER
    # temp_df['previous_inspection_delta'] = temp_df[['restaurant_id', 'inspection_date']].apply(lambda x: delta.loc[x.restaurant_id, x.inspection_date], axis=1)
    # # clean up
    # df.restaurant_id = df.restaurant_id.astype('category')
    # df.drop('temp_date', axis=1, inplace=True)
    # pd.merge resets all the datatypes so doing this instead.
    delta = delta.reset_index()
    delta = delta.rename(columns={'temp_date': 'previous_inspection_delta'})
    delta.previous_inspection_delta = delta.previous_inspection_delta.dt.days
    df = pd.concat([df, pd.merge(temp_df, delta, how='left', on=['restaurant_id', 'inspection_date'])['previous_inspection_delta']], axis=1)

    # transform inspection date
    df['inspection_year'] = df['inspection_date'].dt.year
    df['inspection_month'] = df['inspection_date'].dt.month
    df['inspection_day'] = df['inspection_date'].dt.day
    df['inspection_dayofweek'] = df['inspection_date'].dt.dayofweek
    df['inspection_quarter'] = df['inspection_date'].dt.quarter
    df['inspection_dayofyear'] = df['inspection_date'].dt.dayofyear

    # remove reviews and tips that occur after an inspection
    no_future_mask = df.review_date > df.inspection_date
    df.ix[no_future_mask, ['review_date', 'review_id', 'review_stars', 'review_text', 'user_id', 'review_votes_cool', 'review_votes_funny', 'review_votes_useful', 'user_average_stars', 'user_compliments_cool', 'user_compliments_cute', 'user_compliments_funny', 'user_compliments_hot', 'user_compliments_list', 'user_compliments_more', 'user_compliments_note', 'user_compliments_photos', 'user_compliments_plain', 'user_compliments_profile', 'user_compliments_writer', 'user_fans', 'user_name', 'user_review_count', 'user_votes_cool', 'user_votes_funny', 'user_votes_useful', 'review_year', 'review_month', 'review_day', 'review_dayofweek', 'review_quarter', 'review_dayofyear', 'user_yelping_since_delta', 'user_most_recent_elite_year_delta', 'review_delta']] = np.nan
    # the above is even faster still
    # no_future = lambda x: np.nan if x.review_date > x.inspection_date else x.review_text
    # df.review_text = df.apply(no_future, axis=1)
    # the above maintains all the other information. below gets rid of the entire observation and potentially loses non-review related information if a restaurant is left with no reviews.
    # no_future = features_response[features_response.review_date < features_response.inspection_date]

    # # bin time delta data
    # bin_size = 30
    # tdmax = df.review_delta.max()
    # tdmin = df.review_delta.min()
    # df['review_delta_bin'] = pd.cut(df["review_delta"], np.arange(tdmin, tdmax, bin_size))
    # df['review_delta_bin_codes'] = df.review_delta_bin.astype('category').cat.codes
    # tdmax = df.previous_inspection_delta.max()
    # tdmin = df.previous_inspection_delta.min()
    # df['previous_inspection_delta_bin'] = pd.cut(df["previous_inspection_delta"], np.arange(tdmin-1, tdmax, bin_size))
    # df['previous_inspection_delta_bin_codes'] = df.previous_inspection_delta_bin.astype('category').cat.codes

    return df


def make_train_test():
    # creates hierarchical dataframe with all of the reviews ever given to a restaruant duplicated for every inspection date for a restaurant
    full_features = get_full_features()

    # transform features
    print('transforming features')
    transformed_features = transform_features(full_features)

    # get response
    training_response = pd.read_csv("data/train_labels.csv", index_col=None)
    training_response.columns = ['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=None)
    submission.columns = ['inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']

    # combine features and response
    training_df = make_feature_response(transformed_features, training_response)
    test_df = make_feature_response(transformed_features, submission)

    training_df, test_df = make_categoricals(training_df, test_df)

    training_df = post_transformations(training_df)
    test_df = post_transformations(test_df)

    print('finished transformations')

    # save dataframes
    training_df.to_pickle('pickle_jar/training_df.pkl')
    test_df.to_pickle('pickle_jar/test_df.pkl')
    print('both dataframes pickled')

    # make flat dataframes with one observation per inspection and save them
    print('making flat dataframes')
    print("Response shape of {}".format(training_response.shape))
    print("Submission shape of {}".format(submission.shape))
    flat_train = make_flat_version(training_df)
    flat_test = make_flat_version(test_df)
    flat_train.to_pickle('pickle_jar/flat_train_df.pkl')
    flat_test.to_pickle('pickle_jar/flat_test_df.pkl')

    # save column/feature names since they have grown out of hand
    choices_choices = [str(j)+' - '+str(k) for j, k in zip(training_df.dtypes.index, training_df.dtypes)]
    with open('feature_names.txt', 'w') as f:
        f.write('\n'.join(choices_choices))

    # store = pd.HDFStore('pickle_jar/df_store.h5')
    # store.append('training_df', training_df, data_columns=True, dropna=False)
    # print('training_df in hdfstore')
    # store.append('test_df', test_df, data_columns=True, dropna=False)
    # store.close()


def get_selects(frame, features=None):
    if frame == 'train':
        df = pd.read_pickle('pickle_jar/training_df.pkl')
        if features:
            features = features[:]
            features.extend(['review_delta', 'previous_inspection_delta', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3'])
            return df[features]
        else:
            return df
    elif frame == 'test':
        df = pd.read_pickle('pickle_jar/test_df.pkl')
        if features:
            features = features[:]
            features.extend(['review_delta', 'previous_inspection_delta', 'inspection_id', 'inspection_date', 'restaurant_id', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3'])
            return df[features]
        else:
            return df


def test():
    df = pd.read_pickle('pickle_jar/test_df.pkl')
    store = pd.HDFStore('pickle_jar/df_store.h5')
    store.append('test_df', df, dropna=False, data_columns=['restaurant_id', 'restaurant_full_address', 'review_text', 'user_id'])
    store.close()


def load_dataframes(features=None):
    train_df = get_selects('train', features)
    test_df = get_selects('test', features)
    return train_df, test_df


def get_flats():
    train = pd.read_pickle('pickle_jar/flat_train_df.pkl')
    test = pd.read_pickle('pickle_jar/flat_test_df.pkl')
    return train, test


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
