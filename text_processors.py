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
    # fiting to just the original review corpus before it gets multiplied across all the different inspection dates per restaurant. is this going to skew the results when i transform a corpus larger than the fitted corpus?review_text
    with open('models/reviews_tips_original_text.pkl') as f:
        original_text = pickle.load(f)
    if custom_vec:
        vec = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,3), stop_words='english', lowercase=True, sublinear_tf=True, max_df=1.0, strip_accents='unicode')
    else:
        vec = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=0.1)
    vec = vec.fit(original_text)
    joblib.dump(vec, 'models/tfidf_vectorizer_'+description)

    train_text = data_grab.load_train_df()
    train_docs = vec.transform(train_text)
    joblib.dump(train_docs, 'models/tfidf_train_docs_'+description)
    del train_text, train_docs

    test_text = data_grab.load_text_df()
    test_docs = vec.transform(test_text)
    joblib.dump(test_docs, 'models/tfidf_test_docs_'+description)


def main():
    t0 = time()
    # # get plain tfidf
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_and_save(train_text)

    # get tfidf with custom tokenizer
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_custom_token_and_save(train_text)

    train_df, test_df = data_grab.load_dataframes()
    tfidf_text(train_df, test_df)

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')

if __name__ == '__main__':
    main()
