# coding=utf-8
import argparse
import os
import pickle
import csv


import pandas
from semester_work.Model import *


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Model training application', epilog='Example of usage')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable logging', default=False)

    # Reading args
    parser.add_argument(
        '--src-texts', '-c',
        type=str,
        help='Path to texts',
        dest='c',
        required=True
    )

    parser.add_argument(
        '--o-texts', '-o',
        type=str,
        help='Destenation path',
        dest='o',
        required=True
    )

    parser.add_argument(
        '--lm',
        type=str,
        help='Path to saved model',
        dest='lm',
        required=True
    )

    parser.add_argument(
        '--text-encoding',
        type=str,
        dest='encoding',
        help='Encoding in corpus files',
        default='utf-8'
    )


    args, unparsed = parser.parse_known_args()

    # Enable logging if needed
    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARNING))

    if len(unparsed) > 0:
        logging.warning('Can`t parse these arguments:{0}\n'.format(unparsed))

    # Reading
    texts = []
    logging.info('Reading...')
    if os.access(args.c, os.R_OK):
        file = pandas.read_csv(args.c, encoding=args.encoding, delimiter=' ', header=-1)
        texts = pandas.Series(file[0])
    else:
        logging.FATAL('Нет доступа к файлу')
        exit(-1)

    logging.info('Loading model...')
    model = []
    with open(args.lm, 'rb') as f:
        model = pickle.load(f)

    logging.info('Predicting...')
    pred = model.pipeline_clf.predict(texts)

    logging.info('Writing results...')
    with open(args.o, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in zip(texts, pred):
            if len(row[0]) > 0:
                writer.writerow(row)

    logging.info('Done.')


