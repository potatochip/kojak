import pickle
import data_grab
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def load_matrix(filename):
    with open(filename) as f:
        return pickle.load(f)


def tfidf_and_save(train_text, params=None):
    # create a TfidfVectorizer object with english stop words
    vec = TfidfVectorizer(stop_words='english')
    joblib.dump(vec, 'tfidf_vectorizer_'+str(params))
    # with open('tfidf_vectorizer_'+str(params), 'wb') as f:
    #     pickle.dump(vec, f)
    # create the TfIdf feature matrix from the raw text
    train_tfidf = vec.fit_transform(train_text)
    joblib.dump(train_tfidf, 'tfidf_array_'+str(params))
    # with open('tfidf_array_'+str(params), 'wb') as f:
    #     pickle.dump(train_tfidf, f)
    return vec, train_tfidf


def load_tfidf_matrix(params=None):
    # train_tfidf = load_matrix('tfidf_array_'+str(params))
    train_tfidf = joblib.load('tfidf_array_'+str(params))
    return train_tfidf


def load_tfidf_vectorizer(params=None):
    vec = joblib.load('tfidf_vectorizer_'+str(params))
    # vec = load_matrix('tfidf_vectorizer_'+str(params))
    return vec


def main():
    train_text, test_text = data_grab.load_flattened_reviews()
    vec, train_tfidf = tfidf_and_save(train_text)


if __name__ == '__main__':
    main()
