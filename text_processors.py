import sendMessage
import nltk
import re
import pickle
from progressbar import ProgressBar
from time import time
from nltk.stem.snowball import SnowballStemmer
import data_grab
from textblob import TextBlob
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")


def tokenize(text, spellcheck=True, stem=False, lemmatize=True):
    # lowercase, remove non-alphas and punctuation
    b = TextBlob(text)
    if spellcheck:
        b = b.correct()
    words = b.words.lower()
    if lemmatize:
        words = words.lemmatize()
    tokens = [w for w in words if w.isalpha() and w not in stopwords]

    # letters_only = re.sub("[^a-zA-Z]", " ", text)
    # words = letters_only.lower().split()
    # tokens = [w for w in words if w not in stopwords]
    # if stem:
    #     stems = [stemmer.stem(t) for t in tokens]
    #     tokens = stems

    return tokens


def preprocess_text():
    # grab text
    tokened_list = []
    pbar = ProgressBar(maxval=).start()
    for i, text in enumerate(texts):
        tokens = tokenize(text)
        tokened_list.append(tokens)
        pbar.update(i)
    with open('preprocessed_text.pkl', 'wb') as f:
        pickle.dump(tokened_list, f)
    pbar.finish()


def tfidf_and_save(train_text, params=None):
    if not params: params = 'None'
    vec = TfidfVectorizer(stop_words='english')
    train_tfidf = vec.fit_transform(train_text)
    joblib.dump(vec, 'models/tfidf_vectorizer_'+params)
    joblib.dump(train_tfidf, 'models/tfidf_array_'+params)
    return vec, train_tfidf


def tfidf_custom_token_and_save(train_text, params='spell_lemma'):
    if not params: params = 'None'
    vec = TfidfVectorizer(tokenizer=tokenize)
    train_tfidf = vec.fit_transform(train_text)
    joblib.dump(vec, 'models/tfidf_custom_token_vectorizer_'+params)
    joblib.dump(train_tfidf, 'models/tfidf_custom_token_array_'+params)
    return vec, train_tfidf


def load_tfidf_custom_token_matrix(params=None):
    if not params: params = 'None'
    train_tfidf = joblib.load('models/tfidf_custom_token_array_'+params)
    return train_tfidf


def load_tfidf_custom_token_vectorizer(params=None):
    if not params: params = 'None'
    vec = joblib.load('models/tfidf_custom_token_vectorizer_'+params)
    return vec


def load_tfidf_matrix(params=None):
    if not params: params = 'None'
    train_tfidf = joblib.load('models/tfidf_array_'+params)
    return train_tfidf


def load_tfidf_vectorizer(params=None):
    if not params: params = 'None'
    vec = joblib.load('models/tfidf_vectorizer_'+params)
    return vec


def main():
    t0 = time()
    # # get plain tfidf
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_and_save(train_text)

    # get tfidf with custom tokenizer
    train_text, test_text = data_grab.load_flattened_reviews()
    vec, train_tfidf = tfidf_custom_token_and_save(train_text)

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')

if __name__ == '__main__':
    main()
