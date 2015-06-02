import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def load_matrix(filename):
    with open(filename) as f:
        return pickle.load(f)


def tfidf_and_save(train_text, max_features=5000):
    # create a TfidfVectorizer object with english stop words
    vec = TfidfVectorizer(stop_words='english', max_features=max_features)
    with open('tfidf_vectorizer_'+str(max_features), 'wb') as f:
        pickle.dump(vec, f)

    # create the TfIdf feature matrix from the raw text
    train_tfidf = vec.fit_transform(train_text)
    with open('tfidf_array_'+str(max_features), 'wb') as f:
        pickle.dump(train_tfidf, f)
    return vec, train_tfidf


def load_tfidf_matrix(max_features=5000):
    train_tfidf = load_matrix('tfidf_array_'+str(max_features))
    return train_tfidf
