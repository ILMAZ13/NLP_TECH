import pandas
import argparse
import csv
import os
import logging


class readable_file(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.isfile(prospective_file):
            raise argparse.ArgumentTypeError("readable_file:{0} is not a valid path".format(prospective_file))
        if os.access(prospective_file, os.R_OK):
            setattr(namespace, self.dest, prospective_file)
        else:
            raise argparse.ArgumentTypeError("readable_file:{0} is not a readable file".format(prospective_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This app parse second corpus to certain type')
    parser.add_argument('--source', '-c', action=readable_file, type=str, help='Path to second corpus',  required=True)
    parser.add_argument('--dest', '-o', type=str, help='Destination for parsed corpus',  required=True)
    parser.add_argument('--verbose', '-v', action='store_true', help='Show info logging')
    args, unparsed = parser.parse_known_args()

    logging.basicConfig(level=(logging.INFO if args.verbose else logging.WARNING))

    # Print unparsed args if exists
    if len(unparsed) > 0:
        logging.warning('Can`t parse these arguments:{0}\n'.format(unparsed))

    logging.info('Reading...')
    reviews = pandas.read_json(args.source, lines=True)
    texts = reviews['text']
    positive = reviews['positive']

    logging.info('Computing...')
    classes = []
    for bool_class in positive:
        if bool_class:
            classes.append('positive')
        else:
            classes.append('negative')

    logging.info('Writing...')
    with open(args.dest, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for row in zip(texts, classes):
            if len(row[0]) > 0:
                writer.writerow(row)

    logging.info('Done.')





