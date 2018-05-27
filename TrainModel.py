# coding=utf-8
import argparse
import logging
import os
import texterra
import pickle

import pandas
from Model import *

language = 'russian'
word_type = 'surface_all'
pos_neg = []
te = texterra.API(host='http://localhost:8082/texterra/')

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Model training application', epilog='Example of usage')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable logging', default=False)

    # Reading args
    parser.add_argument(
        '--src-train-texts', '-c',
        type=str,
        help='Path to corpus',
        dest='c',
        required=True
    )

    parser.add_argument(
        '--logs-file', '-l',
        type=str,
        help='Path to train result logs',
        dest='l',
        default='log.log'
    )

    parser.add_argument(
        '--pmi',
        type=str,
        help='Path to save pmi, only if features_c3 enabled',
        dest='pmi',
        default='pmi.csv'
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
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Use hand-crafted features'
    )

    parser.add_argument(
        '--feature_c3',
        type=str,
        choices=['true', 'false'],
        default='false',
        help='Use features from c3'
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
    word_type = args.word_type

    # Enable logging if needed
    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARNING))

    # Print unparsed args if exists
    if len(unparsed) > 0:
        logging.warning('Can`t parse these arguments:{0}\n'.format(unparsed))

    # Reading
    texts = []
    classes = []
    logging.info('Reading...')
    if os.access(args.c, os.R_OK):
        file = pandas.read_csv(args.c, encoding=args.encoding, header=-1)
        texts = pandas.Series(file[0])
        classes = pandas.Series(file[1])
    else:
        logging.FATAL('Нет доступа к файлу')
        exit(-1)

    # Warn if database empty
    if len(texts) == 0:
        logging.ERROR('Нет текста для обучения')
        exit(-1)

    if len(texts) < 100:
        logging.WARNING('Текстов слишком мало < 100')

    # language detection
    logging.info('Language detection...')
    result = te.language_detection(texts[0])
    for item in result:
        if item == "ru":
            language = 'russian'
        if item == 'en':
            language = 'english'
    logging.info('Language detected: {0}'.format(language))

    model = Model(args, language)
    logging.info('Start training...')
    model = model.fit(texts, classes, True)

    # Writing
    logging.info('Writing to file...: {0}'.format(args.o))
    # joblib.dump(model.pipeline_clf, args.o)
    with open(args.o, 'wb') as f:
        pickle.dump(model, f)

    logging.info('Writing Finished. Good job!')



