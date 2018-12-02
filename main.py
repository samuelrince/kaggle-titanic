from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify
from logistic_regression_classifier import *

############################ Load data ###################################"
csv_file_object = csv.reader(open('Data/train.csv', 'rt'))  # Load in the csv file
csv_file_object_test = csv.reader(open('Data/test.csv', 'rt'))  # Load in the csv file
# header = csv_file_object.next()  # Skip the fist line as it is a header
data = []  # Create a variable to hold the data
testData = []

for row in csv_file_object:  # Skip through each row in the csv file,
    r = row[0].split(',')
    data.append(r)  # adding each row to the data variable
X = array(data)  # Then convert from a list to an array.
X = X[1:]   # Skip the first line (because csv_file_object.next() isn't working)

for row in csv_file_object_test:
    r = row[0].split(',')
    testData.append(r)
X_test = testData
X_test = X_test[1:]

y = X[:, 1].astype(int)  # Save labels to y
X_test = array(X_test)
#y_test = X_test[:, 1].astype(int)


X = delete(X, 1, 1)  # Remove survival column from matrix X
#X_test = delete(X_test, 1, 1)

ages = []
ages_test = []
Z = []
Z_test = []

################################ Data preprocessing ##########################################"
for x in X:
    #   Pclass, Name, Sex, Age, SibSp, Parch, Fare
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


#We add the mean age where the data is missing
age = []
for x_test in X_test:
    if x_test[9] != '':
        age.append(x_test[9])
        
mean_age = round(mean(ages), 0)

for x_test in X_test:
    if x_test[9] == '' : x_test[9] = mean_age
        
    
    
    
for x_test in X_test:
    
    #   Pclass, Name, Sex, Age, SibSp, Parch, Fare
    z_test = [float(x_test[1]), x_test[3], x_test[4], x_test[5], float(x_test[6]), float(x_test[7]), float(x_test[9])]

    # Mean of age
    if z_test[3] != '':
        ages_test.append(float(z_test[3]))
        z_test[3] = float(z_test[3])

    # Adapting sex feature
    if z_test[2] == 'male':
        z_test[2] = 0    # Replace 'male' by 0
    else:
        z_test[2] = 1    # Replace 'female' by 1

    '''
        # * Adapting title-status
        # * Mr = 0
        # * Mrs = 1
        # * Miss = 2
        # * Mlle = 3
        # * Master = 4
        # * Dr = 5
    '''
    if 'Mr.' in z_test[1]: z_test[1] = 0
    elif 'Mrs.' in z_test[1]: z_test[1] = 1
    elif 'Miss' in z_test[1]: z_test[1] = 2
    elif 'Mlle' in z_test[1]: z_test[1] = 3
    elif 'Master' in z_test[1]: z_test[1] = 4
    elif 'Dr' in z_test[1]: z_test[1] = 5
    elif 'Jonkheer.' in z_test[1]: z_test[1] = 6
    elif 'the Countess.' in z_test[1]: z_test[1] = 7
    elif 'Capt.' in z_test[1]: z_test[1] = 8
    elif 'Col.' in z_test[1]: z_test[1] = 9
    elif 'Rev.' in z_test[1]: z_test[1] = 10
    elif 'Sir.' in z_test[1]: z_test[1] = 11
    elif 'Lady.' in z_test[1]: z_test[1] = 12
    elif 'Major.' in z_test[1]: z_test[1] = 13
    elif 'Ms.' in z_test[1]: z_test[1] = 14
    elif 'Mme.' in z_test[1]: z_test[1] = 15
    elif 'Don.' in z_test[1]: z_test[1] = 16
    else: z_test[1] = 17

    Z_test.append(z_test)

mean_age_test = round(mean(ages_test), 0)

for z_test in Z_test:
    if z_test[3] == '': z_test[3] = mean_age_test

####################################### Dimensionality reduction ###################################
dim = 4
M = mean(Z, 0)
M_test = mean(Z_test, 0)
C = Z - M
C_test = Z_test - M_test
W = dot(C.T, C)
W_test = dot(C_test.T, C_test)
eigval, eigvec = linalg.eig(W)
eigval_test, eigvec_test = linalg.eig(W_test)
idx = eigval.argsort()[::-1]
idx_test = eigval_test.argsort()[::-1]
eigvec = eigvec[:,idx]
eigvec_test = eigvec_test[:,idx_test]

newData = dot(C, real(eigvec[:,:dim]))
newData_test = dot(C_test, real(eigvec_test[:,:dim]))


# LDA
# W, projected_centroid, X_lda = logistic_regression_classifier(newData, y)
# predictedLabels_LDA = predict(newData_test, projected_centroid, W)
# print('prediction : ', predictedLabels_LDA)

# kf_LDA = cross_validation.KFold(X.shape[0], n_folds=10)
# totalInstances = 0  # Variable that will store the total intances that will be tested
# totalCorrect = 0  # Variable that will store the correctly predicted intances




# for trainIndex, testIndex in kf_LDA:
#     trainSet = newData[trainIndex]
#     testSet = newData[testIndex]
#     trainLabels = y[trainIndex]
#     testLabels = y[testIndex]
#
#     W, projected_centroid, X_lda = logistic_regression_classifier(trainSet, trainLabels)
#     predictedLabels_LDA = predict(testSet, projected_centroid, W)
#
#     correct = 0
#     for i in range(testSet.shape[0]):
#         if predictedLabels_LDA[i] == testLabels[i]:
#             correct += 1
#
#     print('Accuracy: ' + str(float(correct) / testLabels.size))
#     totalCorrect += correct
#     totalInstances += testLabels.size
# print('Total Accuracy: ' + str(totalCorrect / float(totalInstances)))


''' Méthode simpliste de classification'''

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

    # print('Accuracy: ' + str(float(correct) / testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
# print('Total Accuracy: ' + str(totalCorrect / float(totalInstances)))
