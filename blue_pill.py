'''
enter the matrix
'''

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from scipy.sparse import csr_matrix, hstack
import data_grab
from progressbar import ProgressBar


def add_categorical_to_matrix(matrix, df, columns):
    lb = LabelBinarizer(sparse_output=True)
    pbar = ProgressBar(maxval=len(columns)).start()
    # alternative workaround
    lbl_enc = LabelEncoder()
    temp_array = lbl_enc.fit_transform(df[columns[0]])
    columns.pop(0)
    for index, i in enumerate(columns):
        labels = lbl_enc.fit_transform(df[i])
        temp_array = np.vstack([temp_array, labels])
        pbar.update(index)
    onehot = OneHotEncoder().fit_transform(temp_array.T)
    matrix = hstack([matrix, onehot])

    # for index, i in enumerate(columns):
    #     # scikit bug where it doesnt recognize what type of object is being passed. have to explicitly declare it a numpy string type. or call LabelEncoder followed by OneHotEncoder. this is way slower anyway
    #     binarized = lb.fit_transform(np.array(df[i], dtype='|S'))
    #     matrix = hstack([matrix, binarized])
    #     pbar.update(index)

    pbar.finish()
    return matrix


def add_numerical_to_matrix(matrix, df, columns):
    temp_array = np.array(df[columns])
    matrix = hstack([matrix, temp_array])
    # pbar = ProgressBar(maxval=len(columns)).start()
    # for index, i in enumerate(columns):
    #     matrix = hstack([matrix, df[columns]])
    #     pbar.update(index)
    # pbar.finish()
    return matrix


def add_bool_to_matrix(matrix, df, columns):
    # enc_label = LabelEncoder()
    pbar = ProgressBar(maxval=len(columns)).start()
    for index, i in enumerate(columns):
        booled = df[i].astype(int)
        matrix = hstack([matrix, booled])
        # labeled = enc_label.fit_transform(df[i])
        # matrix = hstack([matrix, labeled])
        pbar.update(index)
    pbar.finish()
    return matrix


def special_categories_to_matrix(matrix, df, cats):
    # need to do it like this because pd.merge causes a memory overload
    t0 = pd.get_dummies(df[cats[0]])
    cats.pop(0)
    for i in cats:
        new_dummies = pd.get_dummies(df[i])
        pbar = ProgressBar(maxval=len(new_dummies.columns)).start()
        for index, column in enumerate(new_dummies.columns):
            if column not in t0.columns:
                t0 = pd.concat([t0, new_dummies[column]], axis=1)
            else:
                t0[column] = t0[column] + new_dummies[column]
            pbar.update(index)
        pbar.finish()
    matrix = hstack([matrix, csr_matrix(t0)])
    return matrix

def dummies_to_matrix(matrix, df, columns):
    for column in columns:
        dummies = pd.get_dummies(df[column])
        matrix = hstack([matrix, csr_matrix(dummies)])

def proper_array(x, backfill_size=7):
    encoder_prep = lambda x: cats.index(x)
    temp = map(encoder_prep, x)
    zeros = np.zeros(backfill_size, dtype='int')
    zeros[:len(temp)] = temp
    return zeros

def trimmed_matrix(dropped):
    # create initial matrix, restaurant_id giving too much away and causing overfitting with hierchical data
    print('starting with m0')
    lb = LabelBinarizer(sparse_output=True)
    m = lb.fit_transform(dropped.restaurant_id)
    # m = lb.fit_transform(dropped.restaurant_attributes_accepts_credit_cards)
    print(m.shape)


    # build matrix
    # making nan its own category for categorical
    print("adding categorical to matrix")
    # better for converting numerical or categorical with small attirbutes to categorical
    m = dummies_to_matrix(m, dropped, ['review_stars', 'restaurant_stars', 'restaurant_attributes_price_range', 'restaurant_city', 'inspection_year', 'inspection_month', 'inspection_day', 'inspection_dayofweek', 'inspection_quarter'])
    # better for categorical with large number of attributes
    m = add_categorical_to_matrix(m, dropped, ['restaurant_street',  'restaurant_zipcode'])
    print(m.shape)

    print("adding bool to matrix")
    m = add_bool_to_matrix(m, dropped, ['restaurant_attributes_accepts_credit_cards', 'restaurant_attributes_coat_check', 'restaurant_attributes_good_for_dancing', 'restaurant_attributes_good_for_groups', 'restaurant_attributes_good_for_latenight', 'restaurant_attributes_happy_hour', 'restaurant_attributes_has_tv', 'restaurant_attributes_outdoor_seating', 'restaurant_attributes_take_out',  'restaurant_attributes_takes_reservations', 'restaurant_attributes_waiter_service', 'restaurant_attributes_wheelchair_accessible', 'user_ever_elite'])
    print(m.shape)

    print("adding selected dummies to matrix")
    ambience_dum = pd.get_dummies(dropped.restaurant_ambience)[['divey', 'trendy']]
    alcohol_dum = pd.get_dummies(dropped.restaurant_attributes_alcohol)['full_bar']
    attire_dum = pd.get_dummies(dropped.restaurant_attributes_attire)['dressy']
    noise_dum = pd.get_dummies(dropped.restaurant_attributes_noise_level)['very_loud']
    smoke_dum = pd.get_dummies(dropped.restaurant_attributes_smoking)['outdoor']
    music_dum = pd.get_dummies(dropped.restaurant_music)[['dj', 'live']]
    park_dum = pd.get_dummies(dropped.restaurant_parking)['street']
    wifi_dum = pd.get_dummies(dropped.restaurant_attributes_wifi)['no']
    temp_m = csr_matrix([ambience_dum, alcohol_dum, attire_dum, noise_dum, smoke_dum, music_dum, park_dum, wifi_dum]).T
    m = hstack([m, temp_m])
    print(m.shape)

    print("adding restaurant categories to matrix")
    cats = ['restaurant_category_1', 'restaurant_category_2', 'restaurant_category_3', 'restaurant_category_4', 'restaurant_category_5', 'restaurant_category_6', 'restaurant_category_7']
    m = special_categories_to_matrix(m, dropped, cats)
    print(m.shape)

    print("adding restaurant neighborhoods to matrix")
    cats = ['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3']
    m = special_categories_to_matrix(m, dropped, cats)
    print(m.shape)

    print("adding numerical to matrix")
    m = add_numerical_to_matrix(m, dropped, ['review_votes_cool', 'review_votes_funny', 'review_votes_useful', 'restaurant_latitude', 'restaurant_longitude', 'review_delta', 'previous_inspection_delta', 'polarity', 'subjectivity', 'neg', 'neu', 'pos', 'compound', 'user_yelping_since_delta', 'user_review_count', 'user_average_stars' 'restaurant_hours_friday_close', 'restaurant_hours_friday_open', 'restaurant_hours_monday_close', 'restaurant_hours_monday_open', 'restaurant_hours_saturday_close', 'restaurant_hours_saturday_open', 'restaurant_hours_sunday_close', 'restaurant_hours_sunday_open', 'restaurant_hours_thursday_close', 'restaurant_hours_thursday_open', 'restaurant_hours_tuesday_close', 'restaurant_hours_tuesday_open', 'restaurant_hours_wednesday_close', 'restaurant_hours_wednesday_open'])
    print(m.shape)

    # need to decide whether i just want the first value for the similarity vectors
    m = add_numerical_to_matrix(m, dropped, ['manager', 'supervisor', 'training', 'safety', 'disease', 'ill', 'sick', 'poisoning', 'hygiene', 'raw', 'undercooked', 'cold', 'clean', 'sanitary', 'wash', 'jaundice', 'yellow', 'hazard', 'inspection', 'violation', 'gloves', 'hairnet', 'nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold',])
    print(m.shape)

    # numerical = dropped[['review_votes_cool', 'review_votes_funny', 'review_votes_useful', 'user_average_stars', 'user_compliments_cool', 'user_compliments_cute', 'user_compliments_funny', 'user_compliments_hot', 'user_compliments_list', 'user_compliments_more', 'user_compliments_note', 'user_compliments_photos', 'user_compliments_plain', 'user_compliments_profile', 'user_compliments_writer', 'user_fans', 'user_review_count', 'user_votes_cool', 'user_votes_funny', 'user_votes_useful', 'restaurant_attributes_price_range', 'restaurant_latitude', 'restaurant_longitude', 'restaurant_review_count', 'checkin_counts', 'review_delta', 'previous_inspection_delta', 'polarity', 'subjectivity', 'neg', 'neu', 'pos', 'compound', 'user_yelping_since_delta','manager', 'supervisor', 'training', 'safety', 'disease', 'ill', 'sick', 'poisoning', 'hygiene', 'raw', 'undercooked', 'cold', 'clean', 'sanitary', 'wash', 'jaundice', 'yellow', 'hazard', 'inspection', 'violation', 'gloves', 'hairnet', 'nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold']]

    print("matrix shape of {}".format(m.shape))
    joblib.dump(m, 'pickle_jar/trimmed_matrix')


def specials_matrix(dropped):

    # create initial matrix
    print('starting with m0')
    lb = LabelBinarizer(sparse_output=True)
    m = lb.fit_transform(dropped.restaurant_id)
    print(m.shape)

    print("adding restaurant categories to matrix")
    cats = ['restaurant_category_1', 'restaurant_category_2', 'restaurant_category_3', 'restaurant_category_4', 'restaurant_category_5', 'restaurant_category_6', 'restaurant_category_7']
    m = special_categories_to_matrix(m, dropped, cats)
    print(m.shape)

    print("adding restaurant neighborhoods to matrix")
    cats = ['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3']
    m = special_categories_to_matrix(m, dropped, cats)
    print(m.shape)

    print("matrix shape of {}".format(m.shape))
    joblib.dump(m, 'pickle_jar/specials_matrix')

def test(dropped):
    # create initial matrix
    print('starting with m0')
    lb = LabelBinarizer(sparse_output=True)
    m = lb.fit_transform(dropped.restaurant_id)
    print(m.shape)

    print("adding categorical to matrix")
    m = add_categorical_to_matrix(m, dropped, ['inspection_year', 'inspection_month', 'inspection_day', 'inspection_dayofweek', 'inspection_quarter'])
    print(m.shape)

    print("matrix shape of {}".format(m.shape))
    joblib.dump(m, 'pickle_jar/test_matrix')

def just_categorical(dropped):

    # create initial matrix
    print('starting with m0')
    lb = LabelBinarizer(sparse_output=True)
    m = lb.fit_transform(dropped.restaurant_id)
    print(m.shape)

    # build matrix
    # making nan its own category for categorical
    print("adding categorical to matrix")
    m = add_categorical_to_matrix(m, dropped, ['review_stars', 'user_name', 'restaurant_stars', 'restaurant_attributes_ages_allowed', 'restaurant_attributes_alcohol', 'restaurant_attributes_attire', 'restaurant_attributes_byob_corkage', 'restaurant_attributes_noise_level', 'restaurant_attributes_smoking', 'restaurant_attributes_wifi', 'restaurant_city',  'restaurant_hours_friday_close', 'restaurant_hours_friday_open', 'restaurant_hours_monday_close', 'restaurant_hours_monday_open', 'restaurant_hours_saturday_close', 'restaurant_hours_saturday_open', 'restaurant_hours_sunday_close', 'restaurant_hours_sunday_open', 'restaurant_hours_thursday_close', 'restaurant_hours_thursday_open', 'restaurant_hours_tuesday_close', 'restaurant_hours_tuesday_open', 'restaurant_hours_wednesday_close', 'restaurant_hours_wednesday_open', 'restaurant_ambience', 'restaurant_music', 'restaurant_parking', 'restaurant_street', 'restaurant_zipcode',  'inspection_year', 'inspection_month', 'inspection_day', 'inspection_dayofweek', 'inspection_quarter',])
    print(m.shape)

    print("adding bool to matrix")
    m = add_categorical_to_matrix(m, dropped, ['restaurant_attributes_accepts_credit_cards', 'restaurant_attributes_byob', 'restaurant_attributes_caters', 'restaurant_attributes_coat_check', 'restaurant_attributes_corkage', 'restaurant_attributes_delivery', 'restaurant_attributes_dietary_restrictions_dairy_free', 'restaurant_attributes_dietary_restrictions_gluten_free', 'restaurant_attributes_dietary_restrictions_halal', 'restaurant_attributes_dietary_restrictions_kosher', 'restaurant_attributes_dietary_restrictions_soy_free', 'restaurant_attributes_dietary_restrictions_vegan', 'restaurant_attributes_dietary_restrictions_vegetarian', 'restaurant_attributes_dogs_allowed', 'restaurant_attributes_drive_thr', 'restaurant_attributes_good_for_dancing', 'restaurant_attributes_good_for_groups', 'restaurant_attributes_good_for_breakfast', 'restaurant_attributes_good_for_brunch', 'restaurant_attributes_good_for_dessert', 'restaurant_attributes_good_for_dinner', 'restaurant_attributes_good_for_latenight', 'restaurant_attributes_good_for_lunch', 'restaurant_attributes_good_for_kids', 'restaurant_attributes_happy_hour', 'restaurant_attributes_has_tv', 'restaurant_attributes_open_24_hours', 'restaurant_attributes_order_at_counter', 'restaurant_attributes_outdoor_seating',  'restaurant_attributes_payment_types_amex', 'restaurant_attributes_payment_types_cash_only', 'restaurant_attributes_payment_types_discover', 'restaurant_attributes_payment_types_mastercard', 'restaurant_attributes_payment_types_visa', 'restaurant_attributes_take_out',  'restaurant_attributes_takes_reservations', 'restaurant_attributes_waiter_service', 'restaurant_attributes_wheelchair_accessible', ])
    print(m.shape)

    print("adding restaurant categories to matrix")
    cats = ['restaurant_category_1', 'restaurant_category_2', 'restaurant_category_3', 'restaurant_category_4', 'restaurant_category_5', 'restaurant_category_6', 'restaurant_category_7']
    m = special_categories_to_matrix(m, dropped, cats)
    print(m.shape)

    print("adding restaurant neighborhoods to matrix")
    cats = ['restaurant_neighborhood_1', 'restaurant_neighborhood_2', 'restaurant_neighborhood_3']
    m = special_categories_to_matrix(m, dropped, cats)
    print(m.shape)

    print("matrix shape of {}".format(m.shape))
    joblib.dump(m, 'pickle_jar/categorical_matrix')

def main():
    # get dataframe
    dropped = pd.read_pickle('pickle_jar/final_dropped')

    y = dropped[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
    print("y shape of {}".format(y.shape))
    y.to_pickle('pickle_jar/final_y')

    dropped.drop(['score_lvl_1', 'score_lvl_2', 'score_lvl_3'], axis=1, inplace=True)
    print(dropped.columns.tolist())
    print(dropped.shape)

    full_matrix(dropped)

if __name__ == '__main__':
    main()
