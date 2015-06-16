from progressbar import ProgressBar
from gensim.models import word2vec
import logging
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import string
from textblob import TextBlob


stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = SnowballStemmer("english")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_wordlist(text, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    review_text = re.sub("[^a-zA-Z]", " ", text)
    words = review_text.lower().split()
    if remove_stopwords:
        words = [w for w in words if w not in stopwords]
    return(words)


def review_to_sentences(text, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    raw_sentences = tokenizer.tokenize(text.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences


def get_sentences():
    sentences = []

    print "Parsing sentences from training set"
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print "Parsing sentences from unlabeled set"
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)


def train():
    # gensim expects whole sentences so no stopwords removed and not stemming. might be best to leave numbers in too

    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    model.save('models/word2vec_model')


def vectorized_docs():
    model = word2vec.Word2Vec.load('models/word2vec_model')
    model.fit()
