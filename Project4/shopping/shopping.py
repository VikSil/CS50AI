import csv
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    data = pd.read_csv(filename)

    month_dict = {
        'Jan': 0,
        'Feb': 1,
        'Mar': 2,
        'Apr': 3,
        'May': 4,
        'June': 5,
        'Jul': 6,
        'Aug': 7,
        'Sep': 8,
        'Oct': 9,
        'Nov': 10,
        'Dec': 11,
    }

    # normalise data in accordance to the assignment
    data['Month'] = data['Month'].map(month_dict)
    data['VisitorType'].replace('Returning_Visitor', 1, inplace=True)
    data.loc[data['VisitorType'].ne(1), 'VisitorType'] = 0
    data['Revenue'].replace(False, 0, inplace=True)
    data['Revenue'].replace(True, 1, inplace=True)
    data['Weekend'].replace(False, 0, inplace=True)
    data['Weekend'].replace(True, 1, inplace=True)

    # transform labels column to list
    labels = data['Revenue'].to_list()

    # transfer the rest of data to list of lists
    data.drop(['Revenue'], axis=1, inplace=True)
    evidence = data.values.tolist()

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    print('labels are')
    print(labels)

    print('predictions are')
    print(predictions)

    all_positives = 0
    true_positives = 0

    all_negatives = 0
    true_negatives = 0

    for i in range(len(labels)):

        # count negatives
        if labels[i] == 0:
            all_negatives += 1

            if predictions[i] == 0:
                true_negatives += 1

        # count positives
        else:
            all_positives += 1

            if predictions[i] == 1:
                true_positives += 1

    sensitivity = true_positives / all_positives
    specificity = true_negatives / all_negatives

    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
