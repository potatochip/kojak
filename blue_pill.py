'''
enter the matrix
'''

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.sparse import csr_matrix, hstack


def add_categorical_to_matrix(matrix, df, columns):
    lb = LabelBinarizer(sparse_output=True)
    for i in columns:
        binarized = lb.fit_transform(df[i])
        matrix = hstack([matrix, binarized])
    return matrix

def add_numerical_to_matrix(matrix, df, columns):
    for i in columns:
        matrix = hstack([matrix, df[columns]])
    return matrix

def add_bool_to_matrix(matrix, df, columns):
    enc_label = LabelEncoder()
    for i in columns:
        labeled = enc_label.fit_transform(df[i])
        matrix = hstack([matrix, labeled])
    return matrix

def restaurant_categories_to_matrix(matrix, train, df):
    cats = []
    for i in ['restaurant_category_1',
     'restaurant_category_2',
     'restaurant_category_3',
     'restaurant_category_4',
     'restaurant_category_5',
     'restaurant_category_6',
     'restaurant_category_7']:
        cats.extend(train[i].unique().tolist())
    cats = set(cats)
    cats.remove(np.nan)
    cats = sorted(cats)

    t = df.restaurant_categories.apply(proper_array, args=(7,))

    enc = OneHotEncoder(sparse=True)
    e = enc.fit_transform(np.vstack(t))

    matrix = hstack([matrix, e])
    return matrix

def restaurant_neighborhoods_to_matrix(matrix, train, df):
    cats = []
    for i in ['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3']:
        cats.extend(train[i].unique().tolist())
    cats = set(cats)
    cats.remove(np.nan)
    cats = sorted(cats)

    t = df.restaurant_neighborhoods.apply(proper_array, args=(3,))

    enc = OneHotEncoder(sparse=True)
    e = enc.fit_transform(np.vstack(t))

    matrix = hstack([matrix, e])
    return matrix

def proper_array(x, backfill_size=7):
    encoder_prep = lambda x: cats.index(x)
    temp = map(encoder_prep, x)
    zeros = np.zeros(backfill_size, dtype='int')
    zeros[:len(temp)] = temp
    return zeros


# get dataframe
dropped = pd.read_pickle('pickle_jar/final_dropped')
train = data_grab.get_selects('train')

# create initial matrix
lb = LabelBinarizer(sparse_output=True)
m = lb.fit_transform(dropped.restaurant_id)

##FEATURE SELECTION!!


# build matrix
# making nan its own category for categorical
m = add_categorical_to_matrix(m, dropped, ['review_stars', 'user_name', 'restaurant_stars', 'restaurant_attributes_ages_allowed', 'restaurant_attributes_alcohol', 'restaurant_attributes_attire', 'restaurant_attributes_byob_corkage', 'restaurant_attributes_noise_level', 'restaurant_attributes_smoking', 'restaurant_attributes_wifi', 'restaurant_city',  'restaurant_hours_friday_close', 'restaurant_hours_friday_open', 'restaurant_hours_monday_close', 'restaurant_hours_monday_open', 'restaurant_hours_saturday_close', 'restaurant_hours_saturday_open', 'restaurant_hours_sunday_close', 'restaurant_hours_sunday_open', 'restaurant_hours_thursday_close', 'restaurant_hours_thursday_open', 'restaurant_hours_tuesday_close', 'restaurant_hours_tuesday_open', 'restaurant_hours_wednesday_close', 'restaurant_hours_wednesday_open', 'restaurant_ambience', 'restaurant_music', 'restaurant_parking', 'restaurant_street', 'restaurant_zipcode',  'inspection_year', 'inspection_month', 'inspection_day', 'inspection_dayofweek', 'inspection_quarter',])
m = add_numerical_to_matrix(m, dropped, ['review_votes_cool', 'review_votes_funny', 'review_votes_useful', 'user_average_stars', 'user_compliments_cool', 'user_compliments_cute', 'user_compliments_funny', 'user_compliments_hot', 'user_compliments_list', 'user_compliments_more', 'user_compliments_note', 'user_compliments_photos', 'user_compliments_plain', 'user_compliments_profile', 'user_compliments_writer', 'user_fans', 'user_review_count', 'user_votes_cool', 'user_votes_funny', 'user_votes_useful', 'restaurant_attributes_price_range', 'restaurant_latitude', 'restaurant_longitude', 'restaurant_review_count', 'checkin_counts', 'review_delta', 'previous_inspection_delta', 'polarity', 'subjectivity', 'neg', 'neu', 'pos', 'compound', 'user_yelping_since_delta', ])
# want to make sure that i just want the first value for the similarity vectors
m = add_numerical_to_matrix(m, dropped, ['manager', 'supervisor', 'training', 'safety', 'disease', 'ill', 'sick', 'poisoning', 'hygiene', 'raw', 'undercooked', 'cold', 'clean', 'sanitary', 'wash', 'jaundice', 'yellow', 'hazard', 'inspection', 'violation', 'gloves', 'hairnet', 'nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold',])
m = add_bool_to_matrix(m, dropped, ['restaurant_attributes_accepts_credit_cards', 'restaurant_attributes_byob', 'restaurant_attributes_caters', 'restaurant_attributes_coat_check', 'restaurant_attributes_corkage', 'restaurant_attributes_delivery', 'restaurant_attributes_dietary_restrictions_dairy_free', 'restaurant_attributes_dietary_restrictions_gluten_free', 'restaurant_attributes_dietary_restrictions_halal', 'restaurant_attributes_dietary_restrictions_kosher', 'restaurant_attributes_dietary_restrictions_soy_free', 'restaurant_attributes_dietary_restrictions_vegan', 'restaurant_attributes_dietary_restrictions_vegetarian', 'restaurant_attributes_dogs_allowed', 'restaurant_attributes_drive_thr', 'restaurant_attributes_good_for_dancing', 'restaurant_attributes_good_for_groups', 'restaurant_attributes_good_for_breakfast', 'restaurant_attributes_good_for_brunch', 'restaurant_attributes_good_for_dessert', 'restaurant_attributes_good_for_dinner', 'restaurant_attributes_good_for_latenight', 'restaurant_attributes_good_for_lunch', 'restaurant_attributes_good_for_kids', 'restaurant_attributes_happy_hour', 'restaurant_attributes_has_tv', 'restaurant_attributes_open_24_hours', 'restaurant_attributes_order_at_counter', 'restaurant_attributes_outdoor_seating',  'restaurant_attributes_payment_types_amex', 'restaurant_attributes_payment_types_cash_only', 'restaurant_attributes_payment_types_discover', 'restaurant_attributes_payment_types_mastercard', 'restaurant_attributes_payment_types_visa', 'restaurant_attributes_take_out',  'restaurant_attributes_takes_reservations', 'restaurant_attributes_waiter_service', 'restaurant_attributes_wheelchair_accessible', ])
m = restaurant_categories_to_matrix(m, train, dropped)
m = restaurant_neighborhoods_to_matrix(m, train, dropped)

print("matrix shape of {}".format(m.shape))
y = dropped[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
print("y shape of {}".format(y.shape))
joblib.dump(m, 'pickle_jar/final_matrix_no_feature_selection')
y.to_pickle('pickle_jar/final_y')
