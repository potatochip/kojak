import data_grab
import numpy as np
import logging
import transformations
import text_processors
import test_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


estimator = LinearRegression()
pipeline = Pipeline([('clf', estimator), ])
feature_list = ['time_delta', 'review_text']
transformation_list = [('text_length', transformations.text_to_length)]
# vectorized_docs = text_processors.load_count_docs('test')
vectorized_docs = None
filename = 'test.csv'


LOG_FILENAME = 'fit_submit.log'
logging.basicConfig(filename=LOG_FILENAME, format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def logPrint(message):
    print(message)
    logger.info(message)


def fit_and_submit(train_df, test_df, pipeline, filename):
    if vectorized_docs and feature_list:
        X_train = hstack([vectorized_docs[1], coo_matrix(X_train)])
        logPrint('Matrices combined')
    elif vectorized_docs and not feature_list:
        X_train = vectorized_docs[1]
    elif not vectorized_docs and feature_list:
        pass
    elif not vectorized_docs and not feature_list:
        print('whoops!')

    X_test, y_test = test_model.extract_features(test_df)
    X_train, y_train = test_model.extract_features(train_df)

    # predict the counts for the test set
    s1 = pipeline.fit(X_train[feature_list], y_train['score_lvl_1']).predict(X_test[feature_list])
    s2 = pipeline.fit(X_train[feature_list], y_train['score_lvl_2']).predict(X_test[feature_list])
    s3 = pipeline.fit(X_train[feature_list], y_train['score_lvl_3']).predict(X_test[feature_list])
    predictions = np.dstack((s1, s2, s3))[0]

    # clip the predictions so they are all greater than or equal to zero
    # since we can't have negative counts of violations
    # SHOULD TRY CLIPPING AFTER AVERAGING SCORES ALSO
    predictions = np.clip(predictions, 0, np.inf)

    # averaging by mean, SHOULD TRY ALT METHODS OF GROUPING SCORES TOGETHER
    test_df[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']] = predictions
    submission_scores = test_df.groupby(['restaurant_id', 'inspection_date', 'inspection_id'])['score_lvl_1', 'score_lvl_2', 'score_lvl_3'].mean()
    temp = submission_scores.reset_index().set_index('inspection_id')

    # write the submission file
    new_submission = data_grab.get_submission()
    indexed_prediction = temp.reindex(new_submission.index)
    if new_submission.shape != indexed_prediction.shape:
        logPrint("ERROR: Submission and prediction have different shapes")
    new_submission[['*', '**', '***']] = np.round(indexed_prediction[['score_lvl_1', 'score_lvl_2', 'score_lvl_3']]).astype(np.int8)
    new_submission.to_csv('predictions/'+filename)


def main():
    train_df, test_df = data_grab.load_dataframes(feature_list)
    logPrint("Dataframes loaded")

    # transformations
    trans_list = []
    if transformation_list:
        for title, func in transformation_list:
            trans_list.append(title)
            print("Training set transform")
            train_df = test_model.transform(train_df, func)
            print("Test set transform")
            test_df = test_model.transform(test_df, func)
    logPrint('feature extraction finished')

    if vectorized_docs:
        trans_list.append(vectorized_docs[0])

    fit_and_submit(train_df, test_df, pipeline, filename)


if __name__ == '__main__':
    main()
