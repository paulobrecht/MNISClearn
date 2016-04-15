#===========================================================================
# options

inpath = '/home/pobrecht/Dropbox/ML310/Week1/mnist/data/'
outpath = '/home/pobrecht/Dropbox/ML310/Week1/mnist/'
KsToTry = range(1, 15)



#===========================================================================
# imports

from numpy import ravel, mean, std, cov
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
    """This reads a multi-column integer-only CSV file and assigns its cell values to a list of lists.
    It assumes there are no quoted cells and all commas are field separators.
    """
    x = open(inFile, 'r')

    print("Reading " + inFile + "...")
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

# compute mean, std, and pct of white space within bounding box
def rcSummary(inObj, gridSqDim):
    """This takes in a list of lists, representing a NxN square grid with by-row loading
    and returns a list of tuples: [(row avg 1, col avg 1), (row avg 2, col avg 2), ...]
    """
    flt = float(gridSqDim)
    rowAvg = [] # row average
    colAvg = [] # col average

    i = 0
    rz = []
    for row in inObj: # row avg
        ra = sum(row)/flt
        rowAvg.append(ra) # append to list of row avgs for this grid
        if ra > 0:
            rz.append(i)
        i += 1

    i = 0
    cz = []
    for col in range(0, gridSqDim): # col avg
        ca = sum([inObj[i][col] for i in range(0, gridSqDim)])/flt
        colAvg.append(ca) # append to list of col avgs for this grid
        if ca > 0:
            cz.append(col)
        i += 1

    c1 = cov(rowAvg, colAvg)
    outObj = [float(c1[i][j]) for i in range(2) for j in range(2)]
    cond = [inObj[i][j] for i in rz for j in cz]
    condWhite = cond.count(0)/float(len(cond))

    outObj.append(condWhite) # four covariance matrix entries plus the % empty pixes in the character bounding box
    return outObj

# compute row col averages for each 28x28 image and
# append those averages to the training and validation data
def gridAppend(inObj1, inObj2, gridSqDim):
    """Very special purpose."""
    i = 0
    for grid in inObj1:
        for item in rcSummary(grid, gridSqDim):
            inObj2[i].append(item)
        i += 1

#===========================================================================
# READ DATA

x2 = readCSV(inpath + 'mnist_trn_X.csv') # Read in training design matrix
y2 = readCSV(inpath + 'mnist_trn_y.txt') # Read in training Ys
t2 = readCSV(inpath + 'mnist_tst_X.csv') # Read in validation Xs



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

# model selection on a random 20% sample of the training data
x2shuf = x2shuf[1:len(x2)/5]
y2shuf = y2shuf[1:len(x2)/5]
#x2shuf = x2shuf[1:100] #testing
#y2shuf = y2shuf[1:100] #testing

# 4-fold cross-validation
kf = KFold(len(x2shuf), n_folds=4) # define 4 folds

loop = 0
results = []
wrong = []
for train, test in kf: # iterate over folds
    trainX = [x2shuf[i] for i in train]
    testX = [x2shuf[i] for i in test]
    trainY = ravel([y2shuf[i] for i in train]) 
    testY = ravel([y2shuf[i] for i in test])

    for k in KsToTry:
        knn = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', n_jobs=20)

        # train on 3/4, test on 1/4
        knn.fit(trainX, trainY)
        pred = knn.predict(testX)
        
        # correct prediction rate
        c = 0
        for num in range(len(pred)):
            tup = (loop, k, testY[num], pred[num]) # output incorrect predictions for investigating later
            if pred[num] == testY[num]:
                c += 1
            else:
                wrong.append(tup)

        # save the correct prediction rate to a tuple for evaluation after the loop has completed
        # this should be a dict with k as the key, someday when I have time
        myresult = (k, loop, c, len(pred))
        results.append(myresult) 
        print("Finished Loop " + str(loop+1) + " for K = " + str(k))
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

knn = KNeighborsClassifier(n_neighbors=optimalK, n_jobs=20) # train with optimalK
knn.fit(x2, ravel(y2)) # train on entire training dataset
knn.fit(x2[1:100], ravel(y2[1:100])) # testing
pred = knn.predict(t2) # score the validation data



#===========================================================================
# OUTPUT

z = open(mypath + 'compusrv_submission10k.csv', 'w')
for value in pred:
  print>>z, value

z2 = open(mypath + 'compusrv_submission10k_facts1.txt', 'w')
print>>z2, optimalK

z3 = open(mypath + 'compusrv_submission10k_facts2.txt', 'w')
print>>z3, orderedResults

z3 = open(mypath + 'compusrv_submission10k_facts3.txt', 'w')
print>>z3, wrong
