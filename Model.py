import logging

import nltk
from lightgbm.sklearn import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
# from sklearn.cross_validation import KFold
from analyze_statisitc import generate_auto_lex
from text_process import TextAnalyser


class CustomFeatures(BaseEstimator, TransformerMixin):
    textAnalyser = []

    def __init__(self, text_analyser):
        self.textAnalyser = text_analyser

    def fit(self, x, y=None):
        return self

    def transform(self, text):
        counts = self.textAnalyser.features_list
        self.textAnalyser.features_list = []
        self.textAnalyser.features_names = []
        self.textAnalyser.pos_change = 0
        self.textAnalyser.neg_change = 0
        self.textAnalyser.text = ""
        # last added
        self.textAnalyser.eng_lex_pos = ""
        self.textAnalyser.eng_lex_neg = ""
        self.textAnalyser.eng_lex = ""
        self.textAnalyser.rus_lex = ""
        self.textAnalyser.auto_lex = []
        return counts

    def get_feature_names(self):
        return self.textAnalyser.features_names


class PosTagTransformer(BaseEstimator, TransformerMixin):
    language = 'russian'

    def __init__(self, language):
        if language == 'russian':
            self.language = 'rus'
        else:
            self.language = 'eng'

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        logging.info('POS tagging started...')
        features = []
        for text in texts:
            pos = nltk.tokenize.word_tokenize(text)
            pos = nltk.pos_tag(pos, lang=self.language)
            nn, rb, vb, jj = 0, 0, 0, 0
            for word in pos:
                if 'NN' in word[1]:
                    nn += 1
                elif 'RB' in word[1]:
                    rb += 1
                elif 'VB' in word[1]:
                    vb += 1
                elif 'JJ' in word[1]:
                    jj += 1
            features.append([nn, rb, vb, jj])
        logging.info('POS tagging finished')
        return features

    def get_feature_names(self):
        return ['NN', 'RB', 'VB', 'JJ']


class Model:
    word_type = 'surface_all'
    n = 1
    features = False
    feature_c3 = False
    laplace = False
    unknown_word_freq = 1
    clf = 'svm'
    vectorizer = 'tf_idf'
    pipeline_clf = Pipeline
    textAnalyser = []
    logs_file = 'log.log'
    pmi_file = 'pmi.csv'

    def __init__(self, args, language):
        self.word_type = args.word_type
        self.n = args.n
        self.features = args.features
        self.laplace = args.laplace
        self.unknown_word_freq = args.unknown_word_freq
        self.feature_c3 = args.feature_c3
        self.logs_file = args.l
        self.pmi_file = args.pmi
        if args.clf == 'svm':
            self.clf = SVC()
        elif args.clf == 'logistic_regression':
            self.clf = LogisticRegression()
        elif args.clf == 'naive_bayes':
            self.clf = MultinomialNB()
        elif args.clf == 'lgbm':
            self.clf = LGBMClassifier()
        logging.info("Using Classifier {0}".format(args.clf))

        self.textAnalyser = TextAnalyser(language, args.word_type, args.features, args.feature_c3)

        self.vectorizer = TfidfVectorizer(
            analyzer=self.textAnalyser.text_process,
            min_df=args.unknown_word_freq,
            encoding=args.encoding,
            ngram_range=(args.n, args.n),
            smooth_idf=args.laplace
        )

        if args.features == 'false':
            self.pipeline_clf = Pipeline([
                ('union', FeatureUnion([
                    ('vectorizer', self.vectorizer)
                ])),
                ('clf', self.clf)
            ])
            logging.info("Created classifier without features")
        else:
            self.pipeline_clf = Pipeline([
                ('union', FeatureUnion([
                    ('vectorizer', self.vectorizer),
                    ('pos_tag', PosTagTransformer(language)),
                    ('counts', CustomFeatures(self.textAnalyser))
                ], n_jobs=1)),
                ('clf', self.clf)
            ])
            logging.info("Created classifier WITH features")

    def fit(self, texts, classes, test_needed):
        if test_needed:
            # kf = KFold(len(classes), 4, random_state=13)
            # logging.info('Start testing on 4 folds...')
            # i = 0
            # for train_index, test_index in kf:
            #     i += 1
            #     logging.info("Fold {0}".format(i))
            #     X_train, X_test = texts[train_index], texts[test_index]
            #     y_train, y_test = classes[train_index], classes[test_index]
            X_train, X_test, y_train, y_test = train_test_split(texts, classes, test_size=0.25, random_state=13)
            if self.feature_c3 == 'true':
                logging.info("Starting auto_lex generation...")
                generate_auto_lex(X_train, y_train, self.pmi_file)
                logging.info("Auto_lex generation finished")
                self.textAnalyser.init_c3(self.pmi_file)
            self.pipeline_clf = self.pipeline_clf.fit(X_train, y_train)
            y_pred = self.pipeline_clf.predict(X_test)
            f = open(self.logs_file, 'w+')
            print(classification_report(y_test, y_pred), file=f)
            print(confusion_matrix(y_test, y_pred), file=f)

            coefs = self.pipeline_clf.named_steps['clf'].coef_.tolist()
            names = self.pipeline_clf.named_steps['union'].get_feature_names()
            tab = "    "
            for i in range(len(coefs)):
                if len(coefs) > 1:
                    print(self.pipeline_clf.named_steps['clf'].classes_[i], file=f)
                else:
                    print(self.pipeline_clf.named_steps['clf'].classes_[1], file=f)
                for j in range(20):
                    max_index = coefs[i].index(max(coefs[i]))
                    coefs[i][max_index] = -1
                    try:
                        word = names[max_index]
                        print(tab, word, file=f)
                    except BaseException:
                        print(tab, "cant_get", file=f)
                if len(coefs) == 1:
                    coefs = self.pipeline_clf.named_steps['clf'].coef_.tolist()
                    print(self.pipeline_clf.named_steps['clf'].classes_[0], file=f)
                    for j in range(20):
                        min_index = coefs[i].index(min(coefs[i]))
                        coefs[i][min_index] = 1
                        word = names[min_index]
                        print(tab, word, file=f)

        else:
            self.pipeline_clf = self.pipeline_clf.fit(texts, classes)
            logging.info('Training finished')

        return self
