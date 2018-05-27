import csv
import pandas
import logging

from analyze_methods import getStat, calcPMI


def text_analyze(file):
    logging.info("Reading...")
    file = pandas.read_csv(file, header=-1)
    texts = pandas.Series(file[0])
    classes = pandas.Series(file[1])
    words, words_amount, reviews_count_pos, reviews_count_neg, words_in_pos, words_in_neg, word_occurrence, word_occurrence_pos, word_occurrence_neg = getStat(texts, classes)
    reviews_count = len(texts)
    local_dictionary_size = len(words)
    pmi_pos, pmi_neg = calcPMI(word_occurrence, word_occurrence_pos, word_occurrence_neg, words_in_pos, words_in_neg, local_dictionary_size, words_amount)

    logging.info("Writing statistics...")
    f1 = open('results/analyze_corp_2.txt', 'w+')
    print('number of reviews: ', file=f1)
    print(reviews_count, file=f1)
    print('number of words: ', file=f1)
    print(words_amount, file=f1)
    print('size of dictionary: ', file=f1)
    print(local_dictionary_size, file=f1)
    print('number of positive reviews: ', file=f1)
    print(reviews_count_pos, file=f1)
    print('number of negative reviews: ', file=f1)
    print(reviews_count_neg, file=f1)
    f1.close()

    logging.info("Writing PMI...")
    with open('results/analyze_corp_2_pmi.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(words)):
            writer.writerow([words[i],  pmi_pos[i] + pmi_neg[i]])


def generate_auto_lex(texts75, classes75, pmi_file):
    words, words_amount, reviews_count_pos, reviews_count_neg, words_in_pos, words_in_neg, word_occurrence, word_occurrence_pos, word_occurrence_neg = getStat(texts75, classes75)
    local_dictionary_size = len(words)
    pmi_pos, pmi_neg = calcPMI(word_occurrence, word_occurrence_pos, word_occurrence_neg, words_in_pos, words_in_neg, local_dictionary_size, words_amount)

    logging.info("Writing PMI...")
    with open(pmi_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(words)):
            writer.writerow([words[i],  pmi_pos[i] + pmi_neg[i]])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    text_analyze('corpus2.csv')
