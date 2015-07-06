import sendMessage
import nltk
import re
import cPickle as pickle
from progressbar import ProgressBar
from time import time
from nltk.stem.snowball import SnowballStemmer
import data_grab
import pandas as pd
from textblob import TextBlob
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")


def get_processed_text(df, column, description):
    # returns the processed text paired in a dictionary with its index category number
    processed = load_processed(column, description)
    # with open('pickle_jar/tokenized_'+column+'_'+description) as f:
    #     processed = pickle.load(f)
    codes = df[column].cat.codes
    docs = []
    for i in codes:
        docs.append(processed[i])
    return docs


def text_to_sentiment(df, column):
    # vader sentiment analysis
    pass


def load_processed(column, description):
    with open('pickle_jar/tokenized_'+column+'_'+description) as f:
        temp_list = pickle.load(f)
    return temp_list


def process_text(df, column, description):
    print("Processing {} {}".format(column, description))
    temp_list = []
    pbar = ProgressBar(maxval=len(df[column])).start()
    for index, i in enumerate(df[column]):
    # pbar = ProgressBar(maxval=len(df[column].cat.categories)).start()
    # for index, i in enumerate(df[column].cat.categories):
        temp_list.append(tokenize(i, spell=False, stem=False, lemma=True, lower=True, stop=True))
        pbar.update(index)
    pbar.finish()
    with open('pickle_jar/tokenized_'+column+'_'+description, 'w') as f:
        pickle.dump(temp_list, f)
    print("Pre-token shape of {} and post-token shape of {}.".format(len(df[column].cat.categories), len(temp_list)))


def tokenize(text, spell=False, stem=False, lemma=False, lower=False, stop=False):
    # lowercase, remove non-alphas and punctuation
    b = TextBlob(unicode(text, 'utf8'))

    if spell:
        b = b.correct()
    words = b.words
    if lower:
        words = words.lower()
    if lemma:
        words = words.lemmatize()
    if stem:
        words = [stemmer.stem(w) for w in words]
    if stop:
        tokens = [w.encode('utf-8') for w in words if w.isalpha() and w not in stopwords]
    else:
        tokens = [w.encode('utf-8') for w in words if w.isalpha()]
    # letters_only = re.sub("[^a-zA-Z]", " ", text)
    return tokens


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


def count_text(description='base', custom_vec=False):
    with open('pickle_jar/reviews_tips_original_text.pkl') as f:
        original_text = pickle.load(f)
    if custom_vec:
        vec = CountVectorizer(analyzer=split_into_lemmas)
    else:
        vec = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    vec = vec.fit(original_text)
    joblib.dump(vec, 'pickle_jar/count_vectorizer_'+description)
    print("vectorizing finished")

    # train_text = data_grab.load_df('training_df')
    train_text = pd.read_pickle('pickle_jar/training_df.pkl')
    train_docs = vec.transform(train_text.review_text)
    joblib.dump(train_docs, 'pickle_jar/count_train_docs_'+description)
    del train_text, train_docs
    print("train count matrix created")

    # test_text = data_grab.load_df('test_df')
    test_text = pd.read_pickle('pickle_jar/test_df.pkl')
    test_docs = vec.transform(test_text.review_text)
    joblib.dump(test_docs, 'pickle_jar/count_test_docs_'+description)
    print("test count matrix created")


def tfidf_text(texts, description, custom_vec=False):
    # fiting to just the original review corpus before it gets multiplied across all the different inspection dates per restaurant. is this going to skew the results when i transform a corpus larger than the fitted corpus? doing otherwise gives the reviews for restaurants that get inspected more frequently altered weight
    # with open('pickle_jar/reviews_tips_original_text.pkl') as f:
    #     texts = pickle.load(f)
    if custom_vec:
        vec = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 3), stop_words='english', lowercase=True, sublinear_tf=True, max_df=1.0)
    else:
        vec = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
    vec = vec.fit(texts)
    joblib.dump(vec, 'pickle_jar/tfidf_vectorizer_'+description)
    print("vectorizing finished")

    train_text = data_grab.get_selects('train', [])
    train_docs = vec.transform(train_text.review_text)
    joblib.dump(train_docs, 'pickle_jar/tfidf_train_docs_'+description)
    del train_text, train_docs
    print("train tfidf matrix created")

    # test_text = data_grab.get_selects('train', [])
    # test_docs = vec.transform(test_text.review_text)
    # joblib.dump(test_docs, 'pickle_jar/tfidf_test_docs_'+description)
    # print("test tfidf matrix created")


def load_tfidf_docs(frame='train', description='base'):
    if frame == 'train':
        docs = joblib.load('pickle_jar/tfidf_train_docs_'+description)
    elif frame == 'test':
        docs = joblib.load('pickle_jar/tfidf_test_docs_'+description)
    return ('review_tfidf', docs)


def load_count_docs(frame='train', description='base'):
    if frame == 'train':
        docs = joblib.load('pickle_jar/count_train_docs_'+description)
    elif frame == 'test':
        docs = joblib.load('pickle_jar/count_test_docs_'+description)
    return ('review_bag_of_words', docs)


def main():
    t0 = time()

    # feature_list = ['review_text']
    # train_df = data_grab.get_selects('train', feature_list)
    # process_text(train_df, 'review_text', 'lemma')

    # preprocess flat review text then create tfidf vector
    train, test = data_grab.get_flats()
    # train.review_text = train.review_text.astype('category')
    process_text(train, 'review_text', 'flat_lemma')
    # texts = get_processed_text(train, 'review_text', 'flat_lemma')
    # tfidf_text(texts, 'flat_review_text')

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')


if __name__ == '__main__':
    main()
