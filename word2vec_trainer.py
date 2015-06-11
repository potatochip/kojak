# import modules & set up logging
from progressbar import ProgressBar
import gensim
import logging
# from pymongo.cursor import CursorType
# import pymongo
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import string
from textblob import TextBlob


stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")

# exhaust_cursor = pymongo.cursor.CursorType.EXHAUST
# client = pymongo.MongoClient()
# db = client.hygiene

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# cursor = db.reviews.find({}, {'text': 1, '_id': 0}, cursor_type=exhaust_cursor)


def tokenize(text, stemmed=True):
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = []
    for sentence in sentences:
        letters_only = re.sub("[^a-zA-Z]", " ", sentence)
        words = letters_only.lower().split()
        stopped = [w for w in words if w not in stopwords]
        if stemmed:
            stems = [stemmer.stem(t) for t in stopped]
            stopped = stems
        tokenized_sentences.append(stopped)
    # correct spelling
    # stem or lemmatize?
    # extract named entities
    return tokenized_sentences


def train():
    # text = tokenize(' '.join([i['text'] for i in cursor]))
    model = gensim.models.Word2Vec(text, workers=3, size=200, min_count=2)
    model.save('models/word2vec_model')


def vectorize_docs():
    model = gensim.models.Word2Vec.load('models/word2vec_model')
    model.fit()
