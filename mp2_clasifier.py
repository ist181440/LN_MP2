import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier

#########################################################################



wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
le = preprocessing.LabelEncoder()

#########################################################################

def import_train(path):
    sentences = []
    labels = []
    split = ''

    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
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
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    for sentence in content:
        split = sentence.split('\n')
        sentences.append(split[0])

    with open(path_l) as f_l:
        content_l = f_l.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content_l = [x.strip() for x in content_l]

    for label in content_l:
        split = label.split(' ')
        labels.append(split[0] + " ")

    return sentences,labels


def pre_process(documents_collection):

    documents = []

    for sen in range(0, len(documents_collection)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(documents_collection[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [wordnet_lemmatizer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)

    return documents

def main():

    path_train = "QuestoesConhecidas.txt"
    path_test_sentences = "NovasQuestoes.txt"
    path_test_labels = "NovasQuestoesResultados.txt"

    train_labels, train_sentences  = import_train(path_train)
    test_sentences, test_labels = import_test(path_test_sentences ,path_test_labels)


    targets = []
    for target in train_labels:

        if target not in targets:
            targets.append(str(target))
        else:
            continue

    le.fit(targets)

    train_labels = le.transform(train_labels)
    test_labels = le.transform(test_labels)


    #vectorizer_train = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    #vectorizer =  CountVectorizer(max_df=0.7,stop_words=stopwords.words('english'))
    #vectorizer_test = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))

    train_sentences = pre_process(train_sentences)
    test_sentences = pre_process(test_sentences)




    X_train = vectorizer.fit_transform(train_sentences).toarray()
    X_test = vectorizer.transform(test_sentences).toarray()

    #CLASSIFIERS
    knn = KNeighborsClassifier(n_neighbors=3)
    clf = GaussianNB()
    forest = RandomForestClassifier()
    perceptron = Perceptron()
    tree = DecisionTreeClassifier(criterion='gini',random_state=0)

    model = knn.fit(X_train, train_labels)
    #model = clf.fit(X_train, train_labels)
    #model = forest.fit(X_train, train_labels)
    #model = perceptron.fit(X_train, train_labels)
    #model = tree.fit(X_train, train_labels)


    predY = model.predict(X_test)

    accurracy = accuracy_score(test_labels, predY)
    print(accurracy)

    skplt.metrics.plot_confusion_matrix(test_labels, predY)
    plt.show()

    #print(train_sentences)
    #print(vectorizer.get_feature_names())
    #print(X)
    #print(cosine_similarity(X_train[0],X_train[2])[0][0])

#main()
if __name__ == "__main__":
       main()
