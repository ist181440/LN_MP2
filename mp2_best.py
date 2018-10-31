import re
import nltk
import warnings
import collections
import numpy as np
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.utils import resample
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support

#########################################################################

warnings.filterwarnings("ignore",category=DeprecationWarning)

#########################################################################

def import_train(path):
    sentences = []
    labels = []
    split = ''

    with open(path) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    for sentence in content:
        split = sentence.split('\t')
        sentences.append(split[1])

        labels.append(split[0])

    return labels, sentences


def import_test(path_s,path_l):
    sentences = []
    labels =    []
    split = ''
    with open(path_s) as f:
        content = f.readlines()

    content = [x.strip() for x in content]

    for sentence in content:
        split = sentence.split('\n')
        sentences.append(split[0])

    with open(path_l) as f_l:
        content_l = f_l.readlines()

    content_l = [x.strip() for x in content_l]

    for label in content_l:
        split = label.split(' ')
        labels.append(split[0] + " ")

    return sentences,labels


def pre_process(sentence_collection):
    sentences = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for sen in range(0, len(sentence_collection)):

        sentence = re.sub(r'\W', ' ', str(sentence_collection[sen]))
        sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

        sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)

        sentence = re.sub(r'^b\s+', '', sentence)
        sentence = sentence.lower()

        sentence = sentence.split()
        sentence = [wordnet_lemmatizer.lemmatize(word) for word in sentence]

        sentence = ' '.join(sentence)
        sentences.append(sentence)

    return sentences



def balance_data_aux(df,minority,majority):
    df_upsampled = resample(minority, replace=True, n_samples=77,random_state=42)
    return df_upsampled


def balance_data(df):
    df_new = pd.DataFrame()
    cols = [x for x in range(16)]
    majority = df[df.label==6]

    for col in cols:
        if col!=6:
            minority = df[df.label==col]
            df_new = pd.concat([df_new, balance_data_aux(df,minority,majority)])
        else:
            df_new = pd.concat([df_new, majority])

    return df_new


def create_data_frame(x_train,y_train):

    df = pd.DataFrame()
    df['sentences'] = x_train
    df['label'] = y_train

    return df

def label_traget(train_labels,test_labels):
    lables_train = []
    labels_test = []

    le = preprocessing.LabelEncoder()
    targets = []

    for target in train_labels:

        if target not in targets:
            targets.append(str(target))
        else:
            continue

    le.fit(targets)

    lables_train = le.transform(train_labels)
    labels_test = le.transform(test_labels)

    return lables_train, labels_test

def create_train_test_data(train_sentences, train_labels, test_sentences, test_labels):

    df = pd.DataFrame()
    df_balanced = pd.DataFrame()

    train_sentences = pre_process(train_sentences)
    test_sentences = pre_process(test_sentences)

    vectorizer = TfidfVectorizer(max_features=115, analyzer='word', min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

    df = create_data_frame(train_sentences,train_labels)
    df_balanced = balance_data(df)


    X = df_balanced.iloc[:,0]
    y = df_balanced['label']

    X_train = vectorizer.fit_transform(X).toarray()
    #X_train = vectorizer.fit_transform(train_sentences).toarray()
    X_test = vectorizer.transform(test_sentences).toarray()

    return X_train, X_test, y


def crate_model(X_train, y_train):
    #CLASSIFIERS
    knn = KNeighborsClassifier(n_neighbors=5)
    tree = DecisionTreeClassifier(criterion='gini',random_state=0)
    log = LogisticRegression()

    #model = knn.fit(X_train, y_train)
    #model = tree.fit(X_train, y_train)
    model = log.fit(X_train, y_train)

    return model


def cross_val(model, train_x, train_y, folds=10):
    predY = cross_val_predict(model, train_x, train_y, cv=folds)
    compute_scores(predY, train_y)


def compute_scores(predY, test_labels):
    metrics = precision_recall_fscore_support(test_labels, predY, average='macro')
    accurracy = accuracy_score(test_labels, predY)

    print('\nAccuracy: {}'.format(accurracy))
    print('\nPrecision: {}'.format(metrics[0]))
    print('\nRecall: {}'.format(metrics[1]))

    print('\nF-Scores: {}'.format(metrics[2]) + '\n')
    skplt.metrics.plot_confusion_matrix(test_labels, predY)
    plt.show()


def main():

    path_train = "QuestoesConhecidas.txt"
    path_test_sentences = "NovasQuestoes.txt"
    path_test_labels = "NovasQuestoesResultados.txt"

    train_labels, train_sentences  = import_train(path_train)
    test_sentences, test_labels = import_test(path_test_sentences ,path_test_labels)


    train_labels, test_labels = label_traget(train_labels,test_labels)
    X_train, X_test, train_labels = create_train_test_data(train_sentences,train_labels, test_sentences, test_labels)

    model = crate_model(X_train, train_labels)
    predY = model.predict(X_test)
    compute_scores(predY, test_labels)
    cross_val(model, X_train, train_labels)


if __name__ == "__main__":
       main()
