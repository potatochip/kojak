from gensim.models import Word2Vec
import logging
from textblob import TextBlob
from multiprocessing import Pool
import data_grab
from nltk.corpus import wordnet as wn
from textblob import Word
import cPickle as pickle
from progressbar import ProgressBar
import itertools


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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


def process_text(text):
    b = TextBlob(unicode(text, 'utf8').strip())
    sentences = b.sentences
    process_tags = lambda x: [Word(i[0]).lemmatize(penn_to_wn(i[1])) for i in x.lower().tags]
    return map(process_tags, sentences)


def sentence_pool(df):
    pool = Pool()
    documents = pool.map(process_text, set(df.review_text))
    pool.close()
    pool.join()
    unwind = list(chain.from_iterable(documents))
    no_dupes = {tuple(i) for i in unwind}
    with open('pickle_jar/tokenized_processed_sentences', 'w') as f:
        pickle.dump(no_dupes, f)
    return no_dupes


def get_sentences(df):
    pbar = ProgressBar(maxval=len(set(df.review_text))).start()
    sentence_list = set()
    for index, i in enumerate(set(df.review_text)):
        b = TextBlob(unicode(i, 'utf8').strip())
        sentences = b.sentences
        process_tags = lambda x: [Word(i[0]).lemmatize(penn_to_wn(i[1])) for i in x.lower().tags]
        for i in map(process_tags, sentences):
            sentence_list.add(tuple(i))
        pbar.update(index)
    pbar.finish()
    with open('pickle_jar/tokenized_processed_sentences', 'w') as f:
        pickle.dump(sentence_list, f)
    return sentence_list

# import logging
# import nltk
# import re
# from nltk.stem.snowball import SnowballStemmer
# import string

# stopwords = set(nltk.corpus.stopwords.words('english'))
# stemmer = SnowballStemmer("english")
#
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

def get_sents():
    sentences = []
    train, test = data_grab.get_flats()
    for review in train.review_text:
        sentences += review_to_sentences(review, tokenizer)
    return sentences

def train():
    # gensim expects whole sentences so no stopwords removed
    train, test = data_grab.get_flats()
    sentences = get_sentences(train)

    num_features = 500    # Word vector dimensionality
    min_word_count = 5   # Minimum word count
    num_workers = -1       # Number of threads to run in parallel
    context = 20          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    print "Training model..."
    model = Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count, window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    model.save('pickle_jar/word2vec_model')


def vectorized_docs():
    model = Word2Vec.load('pickle_jar/word2vec_model')
    return model


if __name__ == '__main__':
    train()
