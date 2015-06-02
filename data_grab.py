from sklearn.externals import joblib

import numpy as np
import pandas as pd


def get_reviews():
    '''
    returns reviews remapped to business_ids as a dataframe along with other review information
    '''
    id_map = pd.read_csv("data/restaurant_ids_to_yelp_ids.csv")
    id_dict = {}

    # each Yelp ID may correspond to up to 4 Boston IDs
    for i, row in id_map.iterrows():
        boston_id = row["restaurant_id"]
        # get the non-null Yelp IDs
        non_null_mask = ~pd.isnull(row.ix[1:])
        yelp_ids = row[1:][non_null_mask].values
        for yelp_id in yelp_ids:
            id_dict[yelp_id] = boston_id

    with open("data/yelp_academic_dataset_review.json", 'r') as review_file:
        # the file is not actually valid json since each line is an individual
        # dict -- we will add brackets on the very beginning and ending in order
        # to make this an array of dicts and join the array entries with commas
        review_json = '[' + ','.join(review_file.readlines()) + ']'
    # read in the json as a DataFrame
    reviews = pd.read_json(review_json)

    # replace yelp business_id with boston restaurant_id
    map_to_boston_ids = lambda yelp_id: id_dict[yelp_id] if yelp_id in id_dict else np.nan
    reviews.business_id = reviews.business_id.map(map_to_boston_ids)

    # rename first column to restaurant_id so we can join with boston data
    reviews.columns = ["restaurant_id", "date", "review_id", "stars", "text", "type", "user_id", "votes"]

    # drop restaurants not found in boston data
    reviews = reviews[pd.notnull(reviews.restaurant_id)]
    return reviews


def get_response():
    train_labels = pd.read_csv("data/train_labels.csv", index_col=0)
    # get just the targets from the training labels
    train_targets = train_labels[['*', '**', '***']].astype(np.float64)
    return train_labels, train_targets


def get_submission():
    submission = pd.read_csv("data/SubmissionFormat.csv", index_col=0)
    return submission


def get_tips():
    pass


def get_business_info():
    pass


def get_checkins():
    pass


def get_user_info():
    pass


def get_everything():
    pass


def flatten_reviews(label_df, reviews):
    """
        label_df: inspection dataframe with date, restaurant_id
        reviews: dataframe of reviews

        Returns all of the text of reviews previous to each
        inspection listed in label_df.
    """
    reviews_dictionary = {}
    N = len(label_df)

    for i, (pid, row) in enumerate(label_df.iterrows()):
        # we want to only get reviews for this restaurant that ocurred before the inspection
        pre_inspection_mask = (reviews.date < row.date) & (reviews.restaurant_id == row.restaurant_id)

        # pre-inspection reviews
        pre_inspection_reviews = reviews[pre_inspection_mask]

        # join the text
        all_text = ' '.join(pre_inspection_reviews.text)

        # store in dictionary
        reviews_dictionary[pid] = all_text

        if i % 2500 == 0:
            print '{} out of {}'.format(i, N)

    # return series in same order as the original data frame
    return pd.Series(reviews_dictionary)[label_df.index]


def train_and_save(reviews, train_labels, submission):
    train_text = flatten_reviews(train_labels, reviews)
    test_text = flatten_reviews(submission, reviews)
    joblib.dump(train_text, 'flattened_train_reviews')
    joblib.dump(test_text, 'flattened_test_reviews')
    # with open('flattened_reviews_train_text.pkl', 'wb') as f:
    #     pickle.dump(train_text, f)
    # with open('flattened_reviews_test_text.pkl', 'wb') as f:
    #     pickle.dump(test_text, f)
    return train_text, test_text


def load_flattened_reviews():
    train_text = joblib.load('flattened_train_reviews.pkl')
    test_text = joblib.load('flattened_test_reviews.pkl')
    # with open('flattened_reviews_train_text.pkl') as f:
    #     train_text = pickle.load(f)
    # with open('flattened_reviews_test_text.pkl') as f:
    #     test_text = pickle.load(f)
    return train_text, test_text


def main():
    reviews = get_reviews()
    train_labels, train_targets = get_response()
    submission = get_submission()
    train_and_save(reviews, train_labels, submission)


if __name__ == '__main__':
    main()
