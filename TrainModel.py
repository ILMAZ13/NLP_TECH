# coding=utf-8
import argparse
import logging
import os

import pandas
from text_process import *
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from lightgbm.sklearn import LGBMClassifier

from sklearn.externals import joblib


language = 'russian'
word_type = 'surface_all'

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Model training application', epilog='Example of usage')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable logging', default=False)

    # Reading args
    parser.add_argument(
        '--src-train-texts', '-c',
        type=str,
        help='Path to corpus',
        required=True
    )

    parser.add_argument(
        '--text-encoding',
        type=str,
        dest='encoding',
        help='Encoding in corpus files',
        default='utf-8'
    )

    # Word Type args
    parser.add_argument(
        '--word-type',
        type=str,
        choices=['surface_all', 'surface_no_pm', 'stem'],
        dest='word_type',
        help='In case of "surface_all" – all of tokens take without change\n\t"surface_no_pm" – all tokens, without punctuation\n\t"stem" – stemming',
        default='surface_all'
    )

    parser.add_argument(
        '-n',
        type=int,
        required=True,
        help='n-gramm of words and sentences'
    )

    parser.add_argument(
        '--features',
        type=bool,
        default=False,
        help='Use hand-crafted features'
    )

    parser.add_argument(
        '--laplace',
        action='store_true',
        default=False,
        help='Use Laplace smoothing'
    )

    parser.add_argument(
        '--unknown-word-freq',
        type=int,
        default=1,
        dest='unknown_word_freq',
        help='Minimal frequency to mark word unknown'
    )

    parser.add_argument(
        '-o',
        type=str,
        required=True,
        help='Path to save trained model'
    )

    # Classificators
    parser.add_argument(
        '--clf',
        choices=['svm', 'logistic_regression', 'naive_bayes', 'lgbm'],
        help='Choose one of classifiers',
        default='logistic_regression'
    )

    args, unparsed = parser.parse_known_args()

    # Enable logging if needed
    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARNING))

    # Print unparsed args if exists
    if len(unparsed) > 0:
        logging.warning('Can`t parse these arguments:{0}\n'.format(unparsed))

    # Reading
    texts = []
    classes = []
    logging.info('Reading...')
    if os.path.isfile(args.c):
        if os.access(args.c, os.R_OK):
            file = pandas.read_csv(args.c, encoding=args.encoding)
            texts.append(file[0])
            classes.append(file[1])
        else:
            logging.FATAL('Нет доступа к файлу')
            exit(-1)
    else:
        if os.path.isdir(args.c):
            only_files = [f for f in os.listdir(args.c) if os.path.isfile(os.path.join(args.c, f))]
            for file in only_files:
                file = pandas.read_csv(args.c, encoding=args.encoding)
                texts.append(file[0])
                classes.append(file[1])
        else:
            logging.FATAL('Не верный путь к файлу')
            exit(-1)

    # Warn if database empty
    if len(texts) == 0:
        logging.ERROR('Нет текста для обучения')
        exit(-1)

    if len(texts) < 500:
        logging.WARNING('Текстов слишком мало < 500')

    # Vectoring
    logging.info('Creating Vectorizer...')
    vectorizer = TfidfVectorizer(
        analyzer=text_process,
        min_df=args.unknown_word_freq,
        encoding=args.encoding,
        smooth_idf=args.laplace,
        ngram_range=(args.n, args.n)
    )

    logging.info('Fitting and transforming vectorizer...')
    fitted_vectoriser = vectorizer.fit(texts)
    counts = fitted_vectoriser.transform(texts)

    clf = []
    if args.clf == 'svm':
        clf = SVC()
    elif args.clf == 'logistic_regression':
        clf = LogisticRegression()
    elif args.clf == 'naive_bayes':
        clf = MultinomialNB()
    elif args.clf == 'lgbm':
        clf = LGBMClassifier()

    logging.info("Using Classifier {0}".format(args.clf))

    logging.info('Start training...')
    clf.fit(counts, classes)

    # ToDo: add testing on 4-folds
    # Writing
    logging.info('Writing to file...: {0}'.format(args.o))
    joblib.dump(clf, args.o)

    logging.info('Writing Finished. Good job!')



