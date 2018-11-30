from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify

# Load data
csv_file_object = csv.reader(open('Data/train.csv', 'rt'))  # Load in the csv file
# header = csv_file_object.next()  # Skip the fist line as it is a header
data = []  # Create a variable to hold the data

for row in csv_file_object:  # Skip through each row in the csv file,
    r = row[0].split(',')
    data.append(r)  # adding each row to the data variable
X = array(data)  # Then convert from a list to an array.
X = X[1:]   # Skip the first line (because csv_file_object.next() isn't working)

y = X[:, 1].astype(int)  # Save labels to y

X = delete(X, 1, 1)  # Remove survival column from matrix X

# Data preprocessing
for x in X:
    x = delete(x, 0)    # Remove the Passenger ID
    x = delete(x, 1)    # Remove the first characters of name
    # Adapting sex feature
    if x[2] == 'male':
        x[2] = 0    # Replace 'male' by 0
    else:
        x[2] = 1    # Replace 'female' by 1
    # Adapting title-status



# Initialize cross validation
kf = cross_validation.KFold(X.shape[0], n_folds=10)

totalInstances = 0  # Variable that will store the total intances that will be tested
totalCorrect = 0  # Variable that will store the correctly predicted intances

for trainIndex, testIndex in kf:
    trainSet = X[trainIndex]
    testSet = X[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]

    predictedLabels = classify(trainSet, trainLabels, testSet)

    correct = 0
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1

    print('Accuracy: ' + str(float(correct) / testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print('Total Accuracy: ' + str(totalCorrect / float(totalInstances)))
