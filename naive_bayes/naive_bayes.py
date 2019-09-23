import pandas as pd
import numpy as np
import cv2
import random
import time


from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class_num = 10
feature_len = 784
def binaryzation(img):
	cv_img = img.astype(np.uint8)
	cv2.threshold(cv_img, 50,1,cv2.THRESH_BINARY_INV, cv_img)
	return cv_img

def train(train_set, train_labels):
	prior_probability = np.zeros(class_num)
	conditional_probability = np.zeros((class_num, feature_len, 2))

	for i in range(len(train_labels)):
		img = binaryzation(train_set[i])
		label = train_labels[i]

		prior_probability[label] += 1
		for j in range(feature_len):
			conditional_probability[label][j][img[j]] += 1

	for i in range(class_num):
		for j in range(feature_len):
			pix_0 = conditional_probability[i][j][0]
			pix_1 = conditional_probability[i][j][1]

			probalility_0 = (float(pix_0)/float(pix_0+pix_1))*1000000+1
			probalility_1 = (float(pix_1)/float(pix_0+pix_1))*1000000 + 1

			conditional_probability[i][j][0] = probalility_0
			conditional_probability[i][j][1] = probalility_1
	return prior_probability,conditional_probability


def calculate_probability(img, label,prior_probability):
	probability = int(prior_probability[label])

	for i in range(len(img)):
		probability *= int(conditional_probability[label][i][img[i]])
	return probability

def Predict(testset,prior_probability,conditional_probability):
	predict = []

	for img in testset:
		img = binaryzation(img)
		max_label = 0
		max_probability = calculate_probability(img,0, prior_probability)
		for j in range(1,10):
			probability = calculate_probability(img,j, prior_probability)
			if max_probability < probability:
				max_label = j
				max_probability = probability
		predict.append(max_label)
	return np.array(predict)


if __name__ == '__main__':

	print("start read data")
	time_1 = time.time()
	raw_data = pd.read_csv('../data/train.csv',header=0)
	data = raw_data.values
	imgs = data[0::,1::]
	labels = data[::,0]
	train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
	time_2 = time.time()
	print("read data cost", time_2 - time_1, " second")
	print("Start training")
	prior_probability,conditional_probability = train(train_features,train_labels)
	time_3 = time.time()
	print( "training cost ",time_3 - time_2, " second")
	print("Start predicting")
	test_predict = Predict(test_features,prior_probability,conditional_probability)
	time_4 = time.time()
	print( "predicting cost ",time_4 - time_3," second")
	score = accuracy_score(test_labels,test_predict)
	print("The accuracy score is ", score)

