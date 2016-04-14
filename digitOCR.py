#===========================================================================
# options

mypath = '/home/pobrecht/Dropbox/ML310/Week1/mnist/data/'
KsToTry = range(1, 21)



#===========================================================================
# imports

from numpy import ravel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from random import shuffle, seed



#===========================================================================
# set seed

seed(8675309)



#===========================================================================
# FUNCTIONS

# Read csv matrix into list of lists
def readCSV(inFile):
    """This reads a multi-column integer-only CSV file and assigns its cell values to a list of lists."
    It assumes there are no quoted cells and all commas are field separators.
    """
    x = open(inFile, 'r')

    outObj = []
    line = x.readline()
    while line:
        row = []
        line = line.rstrip('\r\n')
        if line.count(",") > 0 and line.count('"') == 0:
            line = line.split(',') 
        for d in line:
            d = int(d)
            row.append(d)
        outObj.append(row)
        line = x.readline()
    return outObj

# change each row into a grid
def rowToGrid(inObj, rowLen, gridSqDim):
    """This changes a single row to a square grid with by-row loading"""
    outObj = []
    for i in range(len(inObj)):
        grid = []
        for b in range(0, rowLen, gridSqDim):
            e = b + gridSqDim
            grid.append(inObj[i][b:e])
        outObj.append(grid)
    return outObj

# compute row and col averages
def rcAvg(inObj, gridSqDim):
    """This takes in a list of lists, representing a NxN square grid with by-row loading
    and returns a list of tuples: [(row avg 1, col avg 1), (row avg 2, col avg 2), ...]
    """
    flt = float(gridSqDim)
    rowAvg = []
    colAvg = []
    for row in inObj: # row avg
        ra = sum(row)/flt
        rowAvg.append(ra)
    for col in range(0, gridSqDim): # col avg
        ca = sum([inObj[i][col] for i in range(0, gridSqDim)])/flt
        colAvg.append(ca)
    zipped = zip(rowAvg, colAvg) # create list of tuples
    return zipped

# compute row col averages for each 28x28 image and
# append those averages to the training and validation data
def gridAppend(inObj1, inObj2, gridSqDim):
    """Very special purpose."""
    i = 0
    for grid in inObj1:
        for tup in rcAvg(grid, gridSqDim):
            inObj2[i].append(tup[0])
            inObj2[i].append(tup[1])
        i += 1

#===========================================================================
# READ DATA

x2 = readCSV(mypath + 'mnist_trn_X.csv') # Read in training design matrix
y2 = readCSV(mypath + 'mnist_trn_y.txt') # Read in training Ys
t2 = readCSV(mypath + 'mnist_tst_X.csv') # Read in validation Xs



#===========================================================================
# Add columns to each row representing averages
# of the 28 columns and of the 28 rows

x3 = rowToGrid(x2, 784, 28)
gridAppend(x3, x2, 28)

t3 = rowToGrid(t2, 784, 28)
gridAppend(t3, t2, 28)



#===========================================================================
# MODEL SELECTION

# shuffle input data so we can select sequential observations (both for sampling here and KFold() below)
index = range(len(x2))
shuffle(index)
x2shuf = []
y2shuf = []
for i in index:
    x2shuf.append(x2[i])
    y2shuf.append(y2[i])

# model selection on a random 25% sample of the training data
x2shuf = x2shuf[1:len(x2)/4]
y2shuf = y2shuf[1:len(x2)/4]

# 4-fold cross-validation
kf = KFold(len(x2shuf), n_folds=4) # define 4 folds

results = []
loop = 0
for train, test in kf: # iterate over folds
    trainX = [x2shuf[i] for i in train]
    testX = [x2shuf[i] for i in test]
    trainY = ravel([y2shuf[i] for i in train]) 
    testY = ravel([y2shuf[i] for i in test])

    for k in KsToTry:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', n_jobs=16)

        # fit on 3/4, score on 1/4
        knn.fit(trainX, trainY)
        pred = knn.predict(testX)
        
        # correct prediction rate)
        c = 0
        for num in range(len(pred)):
            if pred[num] == testY[num]:
                c += 1

        # save the correct prediction rate to a tuple for evaluation after the loop has completed
        # this should be a dict with k as the key, someday when I have time
        myresult = (k, loop, c, len(pred))
        results.append(myresult) 
    loop += 1

# Collate and report results
collatedResults = {}
for k in KsToTry:
    correct=0
    total=0
    for item in results:
        if k == item[0]:
            correct += item[2]
            total += item[3]
    collatedResults[k] = 100*float(correct)/total

orderedResults = sorted(collatedResults.iteritems(), key=lambda x: x[1], reverse=True)

#===========================================================================
# SCORE THE VALIDATION DATA

optimalK = orderedResults[0][0] # choose the best-performing K

knn = KNeighborsClassifier(n_neighbors=optimalK, n_jobs=16) # train with optimalK
knn.fit(x2, ravel(y2)) # train on entire training dataset
pred = knn.predict(t2) # score the validation data

#===========================================================================
# OUTPUT

z = open(mypath + 'compusrv_submission10k.csv', 'w')
for value in pred:
  print>>z, value

z2 = open(mypath + 'compusrv_submission10k_facts1.csv', 'w')
print>>z2, semiOptimalK

z3 = open(mypath + 'compusrv_submission10k_facts2.csv', 'w')
print>>z3, orderedResults
