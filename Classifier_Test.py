from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from pandas import read_csv
# Visualization tools...
import itertools
import numpy as np
import matplotlib.pyplot as plt


def test_classifier(path, classif="knn", normalization=False):
    """
    Classifier Test.

    Open the data file, train a classifier
    with some of the data, and validate by Testing.
    Normalization is for the visualization part...
    """
    classification = None
    # test individual classifiers
    if classif.lower() == "knn":
        classification = KNeighborsClassifier()
    elif classif.lower() == "naive_bayes" or classif.lower() == "nb":
        classification = GaussianNB()  # with gaussian distribution (not good)
    elif classif.lower() == "mlp" or classif.lower() == "neural_network":
        classification = MLPClassifier()  # multi-layered perceptron
    elif classif.lower() == "svc" or classif.lower() == "svm":
        classification = SVC()
    elif classif.lower() == "decision_tree" or classif.lower() == "dt":
        classification = DecisionTreeClassifier()
    elif classif.lower() == "random_forest" or classif.lower() == "rf":
        classification = RandomForestClassifier()
    elif classif.lower() == "adaboost" or classif.lower() == "ab":
        classification = AdaBoostClassifier()  # another ensemble method(bad)
    elif classif.lower() == "qda":
        classification = QuadraticDiscriminantAnalysis()
    elif classif.lower() == "lda":
        classification = LinearDiscriminantAnalysis()
    else:
        print("Please choose a valid classifier!!!")
        return

    # Set dataset up from file.
    dataset = read_csv(path, header=None,
                       names=['lettr', 'x-box', 'y-box', 'width', 'high',
                              'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar',
                              'xybar', 'x2ybr', 'xy2br', 'x-ege', 'xegvy',
                              'y-ege', 'yegvx'])
    classes = get_target_classes(dataset.loc[:, 'lettr'])
    # Split arrays or matrices into subsets for training & testing.
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.loc[:, 'x-box':'yegvx'], dataset.loc[:, 'lettr'], test_size=.2)
    print("Training and validating classifier:", classif, "...")
    # Classifier training algorithm.
    classification.fit(X_train, y_train)
    # Classify the testing data, using the (now trained) classifier.
    predictions = classification.predict(X_test)
    # Evaluate classifier using metrics...
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    kappa = cohen_kappa_score(y_test, predictions)
    # ...and a confusion matrix
    cmatrix = confusion_matrix(y_test, predictions)
    # Output results...
    print("=======Validation summary=======")
    print("Accuracy: " + str(accuracy) + "\nPrecision: " + str(precision)
          + "\nRecall: " + str(recall) + "\nF1: " + str(
        f1) + "\nCohen's Kappa: " + str(kappa))
    print("Preparing the MATPLOTLIB window...")
    plt.figure()
    plot_confusion_matrix(
        cmatrix, classes, normalize=normalization, title='Confusion matrix')
    print("The confusion matrix is:")
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Greys):
    """
    Print and plot the confusion matrix.

    Normalization [0,1] can be applied by setting `normalize=True`.
    """
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True letter')
    plt.xlabel('Predicted letter')


def get_target_classes(set):
    """Find classes from a set dynamically."""
    classes = []
    for i in set:
        if i not in classes:
            classes.append(i)
    classes.sort()
    return classes
