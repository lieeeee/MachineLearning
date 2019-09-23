# from math import log
# import operator

# def calcShannonEnt(dataset):
# 	numEntries=len(dataSet)
# 	labelCounts = {}
# 	for featVec in dataset:
# 		currentLabel=featVec[-1] 
# 		if currentLabel not in labelCounts.keys():
# 			labelCounts[currentLabel]=0
# 		labelCounts[currentLabel] += 1

# 	shannonEnt = 0.0


# 	for key in labelCounts:
# 		prob = float(labelCounts[key])/numEntries
# 		shannonEnt -= prob * log(prob,2)
# 	return shannonEnt


# def createDataset():
# 	dataSet = [[0, 0, 0, 0, 'no'], 
# 				[0, 0, 0, 1, 'no'], 
# 				[0, 1, 0, 1, 'yes'], 
# 				[0, 1, 1, 0, 'yes'], 
# 				[0, 0, 0, 0, 'no'], 
# 				[1, 0, 0, 0, 'no'], 
# 				[1, 0, 0, 1, 'no'], 
# 				[1, 1, 1, 1, 'yes'], 
# 				[1, 0, 1, 2, 'yes'],
# 				[1, 0, 1, 2, 'yes'], 
# 				[2, 0, 1, 2, 'yes'], 
# 				[2, 0, 1, 1, 'yes'], 
# 				[2, 1, 0, 1, 'yes'], 
# 				[2, 1, 0, 2, 'yes'], 
# 				[2, 0, 0, 0, 'no']]
# 	labels = ['years','job','house','money']
# 	return dataSet, labels

# def chooseBestFeatureToSplit(dataset):
# 	numFeatures = len(dataSet[0]) -1
# 	baseEntroy = calcShannonEnt(dataset)
# 	bestInfoGain = 0.0
# 	bestFeature = -1
# 	for i in range(numFeatures):
# 		feaList = [example[i] for example in dataset]
# 		uniqueVals = set(feaList)
# 		newEntropy = 0.0
# 		for value in uniqueVals:
# 			subDataSet = splitDataSet(dataSet, i, value)
# 			prob = len(subDataSet) / float(len(dataSet))
# 			newEntropy += prob * calcShannonEnt(subDataSet)
# 		infoGain = baseEntroy - newEntropy

# 		print("number %d feature infoGain%.3f" % (i,infoGain))
# 		if(infoGain > bestInfoGain):
# 			bestInfoGain = infoGain
# 			bestFeature = i
# 	return bestFeature

# def splitDataSet(dataSet, axis, value):
# 	retDataSet = []
# 	for featVec in dataSet:
# 		if(featVec[axis] == value):
# 			reducedFeatVect = featVec[:axis]
# 			reducedFeatVect.extend(featVec[axis+1:])
# 			retDataSet.append(reducedFeatVect)
# 	return retDataSet


# if __name__=='__main__':
# 	dataSet, features = createDataset()
# 	# print(dataSet)
# 	print("Best value: " + str(chooseBestFeatureToSplit(dataSet)))
# 	# print(calcShannonEnt(dataSet))

from math import log

def creatDataSet(): 
	dataSet=[[0, 0, 0, 0, 'no'], 
			[0, 0, 0, 1, 'no'], 
			[0, 1, 0, 1, 'yes'], 
			[0, 1, 1, 0, 'yes'], 
			[0, 0, 0, 0, 'no'], 
			[1, 0, 0, 0, 'no'], 
			[1, 0, 0, 1, 'no'], 
			[1, 1, 1, 1, 'yes'], 
			[1, 0, 1, 2, 'yes'], 
			[1, 0, 1, 2, 'yes'], 
			[2, 0, 1, 2, 'yes'], 
			[2, 0, 1, 1, 'yes'], 
			[2, 1, 0, 1, 'yes'], 
			[2, 1, 0, 2, 'yes'], 
			[2, 0, 0, 0, 'no']] 
	labels = ['years','job','house','money']
	return dataSet,labels


def calcShannonEnt(dataSet):
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel=featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1 

	shannonEnt=0.0  
	for key in labelCounts:
		prob=float(labelCounts[key])/numEntries
		shannonEnt-=prob*log(prob,2)
	return shannonEnt

def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt((subDataSet))
		infoGain = baseEntropy - newEntropy
		print("number %d feature infoGain %.3f" % (i,infoGain))

		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
			print(bestFeature)
	return bestFeature


def splitDataSet(dataSet,axis,value): 
	retDataSet=[] 
	for featVec in dataSet: 
		if featVec[axis]==value: 
			reducedFeatVec=featVec[:axis] 
			reducedFeatVec.extend(featVec[axis+1:]) 
			retDataSet.append(reducedFeatVec) 
	return retDataSet


if __name__=='__main__':
	dataSet,features=creatDataSet()
	print("Best value: " + str(chooseBestFeatureToSplit(dataSet)))
