import logging

from time import time
import pandas as pd
import numpy as np

LOG_FILENAME = 'transformation.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def logPrint(message):
    print(message)
    logger.info(message)


def logTime(t0, t1, description):
    logPrint("{} seconds elapsed from start after {}.".format(int(t1 - t0), description))


def logShape(shape1, shape2):
    logPrint('Shape before transformation of {} and after transformation of {}'.format(shape1, shape2))


def text_to_length(df):
    s1 = df.shape
    df.loc[:, 'review_text'] = df.review_text.apply(lambda x: len(x))
    logShape(s1, df.shape)
    return df
