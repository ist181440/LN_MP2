from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        split = label.split('\n')
        labels.append(split[0])

    return sentences,labels




def main():

    path_train = "QuestoesConhecidas.txt"
    path_test_sentences = "NovasQuestoes.txt"
    path_test_labels = "NovasQuestoesResultados.txt"

    train_labels, train_sentences  = import_train(path_train)
    test_sentences, test_labels = import_test(path_test_sentences ,path_test_labels)

    vectorizer_train = TfidfVectorizer()
    vectorizer_test = TfidfVectorizer()

    #vectorizer = CountVectorizer()
    X_train = vectorizer_train.fit_transform(train_sentences)
    X_test  = vectorizer_test.fit_transform(test_sentences)

    #print(train_sentences)
    #print(vectorizer.get_feature_names())
    #print(X)
    print(cosine_similarity(X_train[0],X_train[2])[0][0])

#main()
if __name__ == "__main__":
       main()
