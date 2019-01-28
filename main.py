from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify
from logistic_regression_classifier import *
from kNN import *

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

''' Pour sélectionner les features :
    Pclass : 0,
    Name : 1,
    Sex : 2,
    Age : 3,
    Sibp : 4,
    Parch : 5,
    Fare : 6
'''
featuresSelection= [0,1,2,4]
if (1 in featuresSelection):
    indName = featuresSelection.index(1)
else:
    indName = -1
if (3 in featuresSelection):
    indAge = featuresSelection.index(3)
else:
    indAge = -1
if (2 in featuresSelection):
    indSex = featuresSelection.index(2)
else : 
    indSex = -1
    
def catAge(age):
    if age < 4: #bébé
        return(0.0)
    elif age < 12 : #enfant
        return(1.0)
    elif age < 20 : #adolescent
        return(2.0)
    elif age < 50 : #adulte
        return(3.0)
    else:    #old
        return(4.0)

ageslice = False

################################ Data preprocessing ##########################################"
for x in X:
#    print(x)
    #         Pclass,        Name, Sex, Age,     SibSp,       Parch,       Fare
    zinter = [float(x[1]), x[3], x[4], x[5], float(x[6]), float(x[7]), float(x[9])]

    z=[0]*(len(featuresSelection))       #Selection of wanted features
    for i in range(len(featuresSelection)):
        z[i] = zinter[featuresSelection[i]]
    
    # Mean of age
    if indAge!=-1 and z[indAge] != '' and not ageslice:
        ages.append(float(z[indAge]))
        z[indAge] = float(z[indAge])

    # categorization of age
    if indAge != -1 and ageslice:
        if z[indAge] == '':
            z[indAge] = 3.0
        else :
            z[indAge] = catAge(float(z[indAge]))

#    # Adapting sex feature
    if indSex !=-1 and not ageslice:
        if z[indSex] == 'male':
            z[indSex] = 0    # Replace 'male' by 0
        else:
            z[indSex] = 1    # Replace 'female' by 1

    '''
        # * Adapting title-status
        # * Mr = 0
        # * Mrs = 1
        # * Miss = 2
        # * Mlle = 3
        # * Master = 4
        # * Dr = 5
    '''
    if indName !=-1:
        if 'Mr.' in z[indName]: z[indName] = 0
        elif 'Mrs.' in z[indName]: z[indName] = 1
        elif 'Miss' in z[indName]: z[indName] = 2
        elif 'Mlle' in z[indName]: z[indName] = 3
        elif 'Master' in z[indName]: z[indName] = 4
        elif 'Dr' in z[indName]: z[indName] = 5
        elif 'Jonkheer.' in z[indName]: z[indName] = 6
        elif 'the Countess.' in z[indName]: z[indName] = 7
        elif 'Capt.' in z[indName]: z[indName] = 8
        elif 'Col.' in z[indName]: z[indName] = 9
        elif 'Rev.' in z[indName]: z[indName] = 10
        elif 'Sir.' in z[indName]: z[indName] = 11
        elif 'Lady.' in z[indName]: z[indName] = 12
        elif 'Major.' in z[indName]: z[indName] = 13
        elif 'Ms.' in z[indName]: z[indName] = 14
        elif 'Mme.' in z[indName]: z[indName] = 15
        elif 'Don.' in z[indName]: z[indName] = 16
        else: z[indName] = 17

    Z.append(z)


mean_age = round(mean(ages), 0)

if indAge !=-1 and not ageslice:
    for z in Z:
        if z[indAge] == '': 
            z[indAge] = float(mean_age);
            
        

#We add the mean age where the data is missing
if not ageslice:
    age = []
    for x_test in X_test:
        if x_test[9] != '':
            age.append(x_test[9])
            
    mean_age = round(mean(ages), 0)
    
    for x_test in X_test:
        if x_test[9] == '' : x_test[9] = mean_age
        
    
for x_test in X_test:
    
    #   Pclass, Name, Sex, Age, SibSp, Parch, Fare
#    z_test = [float(x_test[1]), x_test[3], x_test[4], x_test[5], float(x_test[6]), float(x_test[7]), float(x_test[9])]
    zinter_test = [float(x_test[1]), x_test[3], x_test[4], x_test[5], float(x_test[7])]
    
    z_test=[0]*(len(featuresSelection))
    for i in range(len(featuresSelection)):  #Selection of wanted features
        z_test[i] = zinter[featuresSelection[i]]
    
        # categorization of age
    if indAge != -1 and ageslice:
        if z_test[indAge] == '':
            z_test[indAge] = 3.0
        else :
            z_test[indAge] = catAge(float(z_test[indAge]))
            
    # Mean of age
    if indAge!=-1 and z_test[indAge] != '' and not ageslice:
        ages_test.append(float(z_test[indAge]))
        z_test[indAge] = float(z_test[indAge])

    # Adapting sex feature
    if indSex !=-1:
        if z_test[indSex] == 'male':
            z_test[indSex] = 0    # Replace 'male' by 0
        else:
            z_test[indSex] = 1    # Replace 'female' by 1
    #
    '''
         * Adapting title-status
         * Mr = 0
         * Mrs = 1
         * Miss = 2
         * Mlle = 3
         * Master = 4
         * Dr = 5
    '''
    if indName != -1 :
        if 'Mr.' in z_test[indName]: z_test[indName] = 0
        elif 'Mrs.' in z_test[indName]: z_test[indName] = 1
        elif 'Miss' in z_test[indName]: z_test[indName] = 2
        elif 'Mlle' in z_test[indName]: z_test[indName] = 3
        elif 'Master' in z_test[indName]: z_test[indName] = 4
        elif 'Dr' in z_test[indName]: z_test[indName] = 5
        elif 'Jonkheer.' in z_test[indName]: z_test[indName] = 6
        elif 'the Countess.' in z_test[indName]: z_test[indName] = 7
        elif 'Capt.' in z_test[indName]: z_test[indName] = 8
        elif 'Col.' in z_test[indName]: z_test[indName] = 9
        elif 'Rev.' in z_test[indName]: z_test[indName] = 10
        elif 'Sir.' in z_test[indName]: z_test[indName] = 11
        elif 'Lady.' in z_test[indName]: z_test[indName] = 12
        elif 'Major.' in z_test[indName]: z_test[indName] = 13
        elif 'Ms.' in z_test[indName]: z_test[indName] = 14
        elif 'Mme.' in z_test[indName]: z_test[indName] = 15
        elif 'Don.' in z_test[indName]: z_test[indName] = 16
        else: z_test[indName] = 17

    
    Z_test.append(z_test)
#
mean_age_test = round(mean(ages_test), 0)

if indAge != -1 and not ageslice:
    for z_test in Z_test:
        if z_test[indAge] == '': z_test[indAge] = mean_age_test

####################################### Dimensionality reduction ###################################

# perform PCA on both 
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

kf_LDA = cross_validation.KFold(X.shape[0], n_folds=10)
totalInstances = 0  # Variable that will store the total intances that will be tested
totalCorrect = 0  # Variable that will store the correctly predicted intances


for trainIndex, testIndex in kf_LDA:
    trainSet = newData[trainIndex]
    testSet = newData[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]

    W, projected_centroid, X_lda = logistic_regression_classifier(trainSet, trainLabels)
    predictedLabels_LDA = predict(testSet, projected_centroid, W)

    correct = 0
    for i in range(testSet.shape[0]):
        if predictedLabels_LDA[i] == testLabels[i]:
            correct += 1

    print('Accuracy: ' + str(float(correct) / testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print('Total Accuracy: ' + str(totalCorrect / float(totalInstances)))


# Initialize cross validation
print ("Essais de kNN :")


kf = cross_validation.KFold(newData.shape[0], n_folds=10)

totalInstances = 0  # Variable that will store the total intances that will be tested
totalCorrect = 0  # Variable that will store the correctly predicted intances

#print(kNN(newData.shape[0],newData,y,newData_test[4,:]))
for trainIndex, testIndex in kf:
    trainSet = newData[trainIndex]
    testSet = newData[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]

    nbtraindata = trainSet.shape[0]
    k = int(sqrt(nbtraindata))
    predictedLabels = kNNgroup(3, trainSet, trainLabels, testSet)

    correct = 0
    for i in range(testSet.shape[0]):
        if predictedLabels[i] == testLabels[i]:
            correct += 1

    print('Accuracy: ' + str(float(correct) / testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print('Total Accuracy: ' + str(totalCorrect / float(totalInstances)))

''' Méthode simpliste de classification'''

print("classification seulement avec le sexe :")
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
