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

ages = []
Z = []
# Data preprocessing
for x in X:
    #   Survived, Name, Sex, Age, SibSp, Parch, Fare
    z = [float(x[1]), x[3], x[4], x[5], float(x[6]), float(x[7]), float(x[9])]

    # Mean of age
    if z[3] != '':
        ages.append(float(z[3]))
        z[3] = float(z[3])

    # Adapting sex feature
    if z[2] == 'male':
        z[2] = 0    # Replace 'male' by 0
    else:
        z[2] = 1    # Replace 'female' by 1

    '''
        # * Adapting title-status
        # * Mr = 0
        # * Mrs = 1
        # * Miss = 2
        # * Mlle = 3
        # * Master = 4
        # * Dr = 5
    '''
    if 'Mr.' in z[1]: z[1] = 0
    elif 'Mrs.' in z[1]: z[1] = 1
    elif 'Miss' in z[1]: z[1] = 2
    elif 'Mlle' in z[1]: z[1] = 3
    elif 'Master' in z[1]: z[1] = 4
    elif 'Dr' in z[1]: z[1] = 5
    elif 'Jonkheer.' in z[1]: z[1] = 6
    elif 'the Countess.' in z[1]: z[1] = 7
    elif 'Capt.' in z[1]: z[1] = 8
    elif 'Col.' in z[1]: z[1] = 9
    elif 'Rev.' in z[1]: z[1] = 10
    elif 'Sir.' in z[1]: z[1] = 11
    elif 'Lady.' in z[1]: z[1] = 12
    elif 'Major.' in z[1]: z[1] = 13
    elif 'Ms.' in z[1]: z[1] = 14
    elif 'Mme.' in z[1]: z[1] = 15
    elif 'Don.' in z[1]: z[1] = 16
    else: z[1] = 17

    Z.append(z)

mean_age = round(mean(ages), 0)

for z in Z:
    if z[3] == '': z[3] = mean_age


# Dimensionality reduction
dim = 4
M = mean(Z, 0)
C = Z - M
W = dot(C.T, C)
eigval, eigvec = linalg.eig(W)
idx = eigval.argsort()[::-1]
eigvec = eigvec[:,idx]

newData = dot(C, real(eigvec[:,:dim]))

print(newData)

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
