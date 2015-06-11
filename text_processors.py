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


def get_review_text_categories(df):
    # returns the review text paired in a dictionary with its category number
    categories = df.review_text.cat.categories
    df.review_text.cat.codes


def text_to_word2vec(key, df, column):
    processed_text = load_processed(column, description)
    for doc in processed_text:
        for word in doc:
            model.similarity(key)
            #math to combine all the words in a doc
    # map back to categories/expanded rows


def text_to_sentiment(df, column):
    #vader sentiment analysis
    pass


def load_processed(column, description):
    with open('models/tokenized_'+column+'_'+description, 'w') as f:
        temp_list = pickle.load(f)
    return temp_list


def process_text(df, column, description):
    temp_list = []
    pbar = ProgressBar(maxval=df.shape[0]).start()
    for index, i in enumerate(df[column].astype('category').cat.categories):
        # temp_list.append(tokenize(i, spell=False, stem=False, lemma=True, lower=True, stop=True))
        temp_list.append(split_into_lemmas(i))
        pbar.update(index)
    with open('models/tokenized_'+column+'_'+description, 'w') as f:
        pickle.dump(temp_list, f)
    print("Pre-token shape of {} and post-token shape of {}.".format(df[column].shape, len(temp_list)))
    pbar.finish()
    return temp_list


def tokenize(text, spell=False, stem=False, lemma=True, lower=False, stop=False):
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
        tokens = [w for w in words if w.isalpha() and w not in stopwords]
    else:
        tokens = [w for w in words if w.isalpha()]
    # letters_only = re.sub("[^a-zA-Z]", " ", text)
    return tokens


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


def count_text(description='base', custom_vec=False):
    with open('models/reviews_tips_original_text.pkl') as f:
        original_text = pickle.load(f)
    if custom_vec:
        vec = CountVectorizer(analyzer=split_into_lemmas)
    else:
        vec = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    vec = vec.fit(original_text)
    joblib.dump(vec, 'models/count_vectorizer_'+description)
    print("vectorizing finished")

    # train_text = data_grab.load_df('training_df')
    train_text = pd.read_pickle('models/training_df.pkl')
    train_docs = vec.transform(train_text.review_text)
    joblib.dump(train_docs, 'models/count_train_docs_'+description)
    del train_text, train_docs
    print("train count matrix created")

    # test_text = data_grab.load_df('test_df')
    test_text = pd.read_pickle('models/test_df.pkl')
    test_docs = vec.transform(test_text.review_text)
    joblib.dump(test_docs, 'models/count_test_docs_'+description)
    print("test count matrix created")


def tfidf_text(description='base', custom_vec=False):
    # fiting to just the original review corpus before it gets multiplied across all the different inspection dates per restaurant. is this going to skew the results when i transform a corpus larger than the fitted corpus? doing otherwise gives the reviews for restaurants that get inspected more frequently altered weight
    with open('models/reviews_tips_original_text.pkl') as f:
        original_text = pickle.load(f)
    if custom_vec:
        vec = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 3), stop_words='english', lowercase=True, sublinear_tf=True, max_df=1.0)
    else:
        vec = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2)
    vec = vec.fit(original_text)
    joblib.dump(vec, 'models/tfidf_vectorizer_'+description)
    print("vectorizing finished")

    train_text = data_grab.get_selects('train', [])
    train_docs = vec.transform(train_text.review_text)
    joblib.dump(train_docs, 'models/tfidf_train_docs_'+description)
    del train_text, train_docs
    print("train tfidf matrix created")

    test_text = data_grab.get_selects('train', [])
    test_docs = vec.transform(test_text.review_text)
    joblib.dump(test_docs, 'models/tfidf_test_docs_'+description)
    print("test tfidf matrix created")


def load_tfidf_docs(frame='train', description='base'):
    if frame == 'train':
        docs = joblib.load('models/tfidf_train_docs_'+description)
    elif frame == 'test':
        docs = joblib.load('models/tfidf_test_docs_'+description)
    return ('review_tfidf', docs)


def load_count_docs(frame='train', description='base'):
    if frame == 'train':
        docs = joblib.load('models/count_train_docs_'+description)
    elif frame == 'test':
        docs = joblib.load('models/count_test_docs_'+description)
    return ('review_bag_of_words', docs)


def main():
    t0 = time()
    # # get plain tfidf
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_and_save(train_text)

    # get tfidf with custom tokenizer
    # train_text, test_text = data_grab.load_flattened_reviews()
    # vec, train_tfidf = tfidf_custom_token_and_save(train_text)

    feature_list = ['review_text']
    train_df, test_df = data_grab.load_dataframes(feature_list)
    process_text(train_df, 'review_text', 'train_lemma')
    process_text(test_df, 'review_text', 'test_lemma')

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')


if __name__ == '__main__':
    main()
