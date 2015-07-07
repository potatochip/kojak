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
from nltk.corpus import wordnet as wn
from textblob import Word
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool

stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")



def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


def preprocess_pool(df):
    df = df[['restaurant_id', 'inspection_date', 'inspection_id', 'review_text', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
    pool = Pool()
    df['preprocessed_review_text'] = pool.map(combine_preprocess, df.review_text)
    pool.close()
    pool.join()
    df.to_pickle('pickle_jar/preprocessed_review_text_df')


def combine_preprocess(text):
    b = TextBlob(unicode(text, 'utf8').strip())
    tags = b.tags
    tokens = map(preprocess, tags)
    tokens = filter(None, tokens)
    return ' '.join(tokens)


def preprocess(tagged):
    word = Word(tagged[0])
    if word.isalpha() and word not in stopwords:
        tag = penn_to_wn(tagged[1])
        l = word.lemmatize(tag)
    else:
        l = ''
    return l.lower()


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


def sentiments(text):
    b = TextBlob(unicode(text, 'utf8').strip())
    # returns the sentiment for all of the reviews together
    sentiment = tuple(b.sentiment)

    # # returns the sentiment of each sentence
    # return map(sentiment, b.sentences)
    # sentiment = lambda x: tuple(x.sentiment)

    return sentiment


def sentiment_pool(df):
    df = df[['restaurant_id', 'inspection_date', 'inspection_id', 'review_text', 'score_lvl_1', 'score_lvl_2', 'score_lvl_3']]
    pool = Pool()
    df['sentiment'] = pool.map(sentiments, df.review_text)
    pool.close()
    pool.join()
    df.to_pickle('pickle_jar/review_text_sentiment_for_all_reviews_df')


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

    # # ngrams
    # temp_list = []
    # for i in range(1,ngram+1):
    #     temp = [list(i) for i in TextBlob(' '.join(tokens)).ngrams(i)]
    #     try:
    #         if len(temp[0]) == 1:
    #             temp_list.extend([i[0] for i in temp])
    #         else:
    #             for i in temp:
    #                 temp_list.append(tuple(i))
    #     except:
    #         pass
    # return temp_list
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
    with open('pickle_jar/reviews_tips_original_text.pkl') as f:
        texts = pickle.load(f)
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

    test_text = data_grab.get_selects('train', [])
    test_docs = vec.transform(test_text.review_text)
    joblib.dump(test_docs, 'pickle_jar/tfidf_test_docs_'+description)
    print("test tfidf matrix created")


def tfidf_flat():
    prep = pd.read_pickle('pickle_jar/preprocessed_review_text_df')
    vec = TfidfVectorizer(ngram_range=(1,3), lowercase=False, sublinear_tf=True, max_df=0.9, min_df=2, max_features=1000)
    tfidf = vec.fit_transform(prep.preprocessed_review_text)
    joblib.dump(tfidf, 'pickle_jar/tfidf_preprocessed_ngram3_sublinear_1mil')


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
    # preprocess_pool(train)
    # tfidf_flat()

    sentiment_pool(train)

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')


if __name__ == '__main__':
    main()
