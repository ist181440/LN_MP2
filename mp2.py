import re
import sys
import nltk
import warnings
import numpy as np

import scikitplot as skplt
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn import preprocessing

from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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


def label_traget(train_labels,test_labels, label_encoder):
    lables_train = []
    labels_test = []

    targets = []

    for target in train_labels:

        if target not in targets:
            targets.append(str(target))
        else:
            continue

    label_encoder.fit(targets)

    lables_train = label_encoder.transform(train_labels)
    labels_test = label_encoder.transform(test_labels)

    return lables_train, labels_test, label_encoder

def create_train_test_data(train_sentences, train_labels, test_sentences, test_labels):

    train_sentences = pre_process(train_sentences)
    test_sentences = pre_process(test_sentences)

    vectorizer = TfidfVectorizer(max_features=115, analyzer='word', min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

    X_train = vectorizer.fit_transform(train_sentences).toarray()
    X_test = vectorizer.transform(test_sentences).toarray()

    return X_train, X_test


def crate_model(X_train, y_train):
    #CLASSIFIERS
    knn = KNeighborsClassifier(n_neighbors=3)
    tree = DecisionTreeClassifier(criterion='gini',random_state=0)
    log = LogisticRegression(random_state=0, solver='lbfgs')
    forest = RandomForestClassifier()
    nbg = GaussianNB()
    nbm = MultinomialNB()

    model = knn.fit(X_train, y_train)
    #model = tree.fit(X_train, y_train)
    #model = log.fit(X_train, y_train)
    #model = forest.fit(X_train, y_train)
    #model = nb.fit(X_train, y_train)
    #model = nbm.fit(X_train, y_train)


    return model

def compute_scores(predY, test_labels):
    metrics = precision_recall_fscore_support(test_labels, predY, average='macro')
    accurracy = accuracy_score(test_labels, predY)

    print('\nAccuracy: {}'.format(accurracy))
    print('\nPrecision: {}'.format(metrics[0]))
    print('\nRecall: {}'.format(metrics[1]))

    print('\nF-Scores: {}'.format(metrics[2]) + '\n')
    skplt.metrics.plot_confusion_matrix(test_labels, predY)

    plt.show()


def get_labels_names(numeric_labels, label_encoder):
    return label_encoder.inverse_transform(numeric_labels)


def find_best_k(X_train, y_train, X_test, test_labels):

    kk = [x for x in range(1,16,2)]
    scores = []

    for k in kk:
        knn = KNeighborsClassifier(n_neighbors=k)
        model = knn.fit(X_train, y_train)
        predy = model.predict(X_test)

        scores.append(accuracy_score(test_labels, predy))

    plt.title("Find Best K")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.xticks(kk)
    plt.plot(kk,scores)
    plt.show()


def compute_max_label(lst, labels_target):
    labels = []
    count = {}

    for el in lst:
        labels.append(labels_target[el[0]])
    count = Counter(labels)

    return count.most_common(1)[0][0]

def compute_cosine_similarit(train, test, train_target):
    predY = []

    similarity = cosine_similarity(test,train)

    for el in similarity:
        maximum = np.argwhere(el == np.amax(el))

        if len(maximum) > 1:
            predY.append(compute_max_label(maximum, train_target))
        else:
            predY.append(train_target[maximum[0][0]])

    return predY


def main():

    args = sys.argv

    path_train = args[1]
    path_test_sentences = args[2]


    path_test_labels = "NovasQuestoesResultados.txt"

    train_labels, train_sentences  = import_train(path_train)
    test_sentences, test_labels = import_test(path_test_sentences ,path_test_labels)

    le = preprocessing.LabelEncoder()

    train_labels, test_labels, le = label_traget(train_labels,test_labels, le)
    X_train, X_test = create_train_test_data(train_sentences,train_labels, test_sentences, test_labels)

    model = crate_model(X_train, train_labels)

    predY = model.predict(X_test)
    #predY = compute_cosine_similarit(X_train, X_test, train_labels)

    #find_best_k(X_train, train_labels, X_test, test_labels)
    #compute_scores(predY, test_labels)

    predY_nominal_labels = get_labels_names(predY, le)

    for el in predY_nominal_labels:
        sys.stdout.write(el + '\n')
    sys.exit(0)



main()
