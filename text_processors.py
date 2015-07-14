import sendMessage
import nltk
import re
import cPickle as pickle
from progressbar import ProgressBar
from time import time
from nltk.stem.snowball import SnowballStemmer
import data_grab
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from textblob import Word
from time import time
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool
from gensim.models import Word2Vec
import numpy as np
from sklearn.externals import joblib
from scipy.sparse import csr_matrix, hstack



# from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
# above breaks ipython print statement so turning it off until needed

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


def preprocess_pool(df, filename):
    pool = Pool()
    # df['preprocessed_review_text'] = pool.map(combine_preprocess, df.review_text.fillna(''))
    df['preprocessed_review_text'] = pool.map(combine_preprocess, df.review_text.dropna())
    pool.close()
    pool.join()
    df.drop('review_text', axis=1, inplace=True)
    df.to_pickle('pickle_jar/'+filename)
    return df


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


def similarity_vector(text):
    b = TextBlob(text)
    similarities = []
    for word in b.words:
        try:
            similarities.append(g_model.similarity(topic, word))
        except:
            try:
                # since google did a shitty preprocessing job and the model is case-sensitive
                similarities.append(g_model.similarity(topic, word.title()))
            except:
                pass
    # keep just the top 100 similar words in reveiw
    sim_vec = np.array(sorted(similarities, reverse=True)[:100], dtype='float32')
    # sim_vec = sorted(similarities, reverse=True)[:100]
    # fill the review with zeroes if its less than 100 words
    backfill = lambda x: np.append(x, np.zeros((100 - len(x),)))
    # backfill = lambda x: list(np.append(x, np.zeros((100 - len(x),))))
    # backfill = lambda x: x + [0]*(100-len(x))
    return backfill(sim_vec) if len(sim_vec) > 0 else np.nan


def similarity_matrix():
    df = pd.read_pickle('pickle_jar/similarity_vectors_df')
    topics = ['supervisor', 'training', 'safety', 'disease', 'ill', 'sick', 'poisoning', 'hygiene', 'raw', 'undercooked', 'cold', 'clean', 'sanitary', 'wash', 'jaundice', 'yellow', 'hazard', 'inspection', 'violation', 'gloves', 'hairnet', 'nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold', ]

    df = df[['manager']+topics]
    df = df.dropna()
    pbar = ProgressBar(maxval=len(topics)).start()
    matrix = csr_matrix(np.vstack(df.manager))
    for index, i in enumerate(topics):
        t = np.vstack(df[i])
        matrix = hstack([matrix, t])
        pbar.update(index)
    pbar.finish()
    joblib.dump(matrix, 'pickle_jar/similarity_matrix')


def similarity_pool(df):
    # save processing time since so many duplicate reviews. just process each identical text once through categories
    df = df.dropna(subset=['preprocessed_review_text'])

    cats = df.preprocessed_review_text.astype('category').cat

    topic_list = ['manager', 'supervisor', 'training', 'safety', 'disease', 'ill', 'sick', 'poisoning', 'hygiene', 'raw', 'undercooked', 'cold', 'clean', 'sanitary', 'wash', 'jaundice', 'yellow', 'hazard', 'inspection', 'violation', 'gloves', 'hairnet', 'nails', 'jewelry', 'sneeze', 'cough', 'runny', 'illegal', 'rotten', 'dirty', 'mouse', 'cockroach', 'contaminated', 'gross', 'disgusting', 'stink', 'old', 'parasite', 'reheat', 'frozen', 'broken', 'drip', 'bathroom', 'toilet', 'leak', 'trash', 'dark', 'lights', 'dust', 'puddle', 'pesticide', 'bugs', 'mold']
    for i in topic_list:
        t0 = time()
        global topic
        topic = i
        print("Working on '{}'".format(topic))
        pool = Pool()
        # df[topic] = pool.map(similarity_vector, df.preprocessed_review_text)
        temp = pool.map(similarity_vector, cats.categories)
        pool.close()
        pool.join()
        docs = []
        for i in cats.codes:
            docs.append(temp[i])
        df[topic] = docs
        print("{} seconds passed".format(time()-t0))
    df.to_pickle('pickle_jar/similarity_vectors_df')


def sentiments(text):
    if pd.isnull(text):
        pass
    else:
        b = TextBlob(unicode(text, 'utf8').strip())
        # returns the sentiment for all of the reviews together
        sentiment = tuple(b.sentiment)

        # # returns the sentiment for each sentence
        # return map(sentiment, b.sentences)
        # sentiment = lambda x: tuple(x.sentiment)

        return sentiment


def sentiment_pool(df, filename):
    pool = Pool()
    df['sentiment'] = pool.map(sentiments, df.review_text)
    df['vader'] = pool.map(vader, df.review_text)
    pool.close()
    pool.join()
    # df.drop('review_text', axis=1, inplace=True)
    df['polarity'] = df.sentiment.apply(lambda x: x if pd.isnull(x) else x[0])
    df['subjectivity'] = df.sentiment.apply(lambda x: x if pd.isnull(x) else x[1])
    df['neg'] = df.vader.apply(lambda x: x if pd.isnull(x) else x['neg'])
    df['neu'] = df.vader.apply(lambda x: x if pd.isnull(x) else x['neu'])
    df['pos'] = df.vader.apply(lambda x: x if pd.isnull(x) else x['pos'])
    df['compound'] = df.vader.apply(lambda x: x if pd.isnull(x) else x['compound'])
    df.to_pickle('pickle_jar/'+filename)


def vader(text):
    '''vader sentiment analysis'''
    if pd.isnull(text):
        pass
    else:
        return vaderSentiment(text)


def load_processed(column, description):
    with open('pickle_jar/tokenized_'+column+'_'+description) as f:
        temp_list = pickle.load(f)
    return temp_list


def process_text(df, column, description):
    print("Processing {} {}".format(column, description))
    temp_list = []
    pbar = ProgressBar(maxval=len(df[column].cat.categories)).start()
    for index, i in enumerate(df[column].cat.categories):
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


def tfidf(df, filename):
    vec = TfidfVectorizer(ngram_range=(1,3), lowercase=False, sublinear_tf=True, max_df=0.9, min_df=2, max_features=1000000)
    texts = df.preprocessed_review_text.replace('', np.nan).dropna()
    print(texts.shape)
    tfidf = vec.fit_transform(texts)
    joblib.dump(tfidf, 'pickle_jar/'+filename)


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

    # preprocess flat review text then create tfidf vector and sentiments
    # train, test = data_grab.get_flats()
    # preprocess_pool(train, 'preprocessed_review_text_flat_df')
    # sentiment_pool(train, 'review_text_sentiment_for_all_reviews_df')
    # prep = pd.read_pickle('pickle_jar/preprocessed_review_text_flat_df')
    # tfidf_flat(prep, 'tfidf_preprocessed_ngram3_sublinear_1mil')


    # preprocess hierchical review text then create tfidf vector
    train = data_grab.get_selects('train')
    prep = preprocess_pool(train, 'preprocessed_review_text_hierarchical_df_dropna')
    del train
    # sentiment_pool(train, 'review_text_sentiment_hierarchical_df')

    # tfidf might be a bit messed up since hierchical is going to make multiples of everything. so places that have more inspection dates are going to end up with reviews that have less weighted text
    # prep = pd.read_pickle('pickle_jar/preprocessed_review_text_hierarchical_df_dropna')
    tfidf(prep, 'tfidf_preprocessed_ngram3_sublinear_1mil_hierarchical_dropna')

    # # create word2vec sentiment vectors
    # prep = pd.read_pickle('pickle_jar/preprocessed_review_text_hierarchical_df')
    # global g_model
    # g_model = Word2Vec.load_word2vec_format('w2v data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    # similarity_pool(prep)

    # similarity_matrix()

    t1 = time()
    print("{} seconds elapsed.".format(int(t1 - t0)))
    sendMessage.doneTextSend(t0, t1, 'text_processors')


if __name__ == '__main__':
    main()
