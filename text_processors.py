import sendMessage
import nltk
import re
import cPickle as pickle
from progressbar import ProgressBar
from time import time
from nltk.stem.snowball import SnowballStemmer
import data_grab
from textblob import TextBlob
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")


def tokenize(text, spellcheck=False, stem=False, lemmatize=True, lowercase=False, stopwords=False):
    # lowercase, remove non-alphas and punctuation
    b = TextBlob(text)
    if spellcheck:
        b = b.correct()
    words = b.words
    if lowercase:
        words = words.lower()
    if lemmatize:
        words = words.lemmatize()
    if stopwords:
        tokens = [w for w in words if w.isalpha() and w not in stopwords]
    else:
        tokens = [w for w in words if w.isalpha()]

    # letters_only = re.sub("[^a-zA-Z]", " ", text)
    # words = letters_only.lower().split()
    # tokens = [w for w in words if w not in stopwords]
    # if stem:
    #     stems = [stemmer.stem(t) for t in tokens]
    #     tokens = stems

    return tokens


def preprocess_text(df, filename):
    tokened_list = []
    pbar = ProgressBar(maxval=df.shape[0]).start()
    for i, text in enumerate(df.review_text):
        tokens = tokenize(text)
        tokened_list.append(tokens)
        pbar.update(i)
    with open('models/'+filename, 'w') as f:
        pickle.dump(tokened_list, f)
    pbar.finish()


def tfidf_text(description='base', custom_vec=False):
    # fiting to just the original review corpus before it gets multiplied across all the different inspection dates per restaurant. is this going to skew the results when i transform a corpus larger than the fitted corpus? doing otherwise gives the reviews for restaurants that get inspected more frequently altered weight
    with open('models/reviews_tips_original_text.pkl') as f:
        original_text = pickle.load(f)
    if custom_vec:
        vec = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,3), stop_words='english', lowercase=True, sublinear_tf=True, max_df=1.0)
    else:
        vec = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
    vec = vec.fit(original_text)
    joblib.dump(vec, 'models/tfidf_vectorizer_'+description)
    print("vectorizing finished")

    train_text = data_grab.load_train_df()
    train_docs = vec.transform(train_text.review_text)
    joblib.dump(train_docs, 'models/tfidf_train_docs_'+description)
    del train_text, train_docs
    print("train tfidf matrix created")

    test_text = data_grab.load_test_df()
    test_docs = vec.transform(test_text.review_text)
    joblib.dump(test_docs, 'models/tfidf_test_docs_'+description)
    print("test tfidf matrix created")


def load_tfidf_matrix(train=True, description='base'):
    if train:
        tfidf_matrix = joblib.load('models/tfidf_train_docs_'+description)
    else:
        tfidf_matrix = joblib.load('models/tfidf_test_docs_'+description)
    return tfidf_matrix


def main():
    t0 = time()
    # # get plain tfidf
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_and_save(train_text)

    # get tfidf with custom tokenizer
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_custom_token_and_save(train_text)

    tfidf_text()

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')

if __name__ == '__main__':
    main()
