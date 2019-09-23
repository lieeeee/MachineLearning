import cv2
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

k = 10
def get_hog_feature(trainset):
	features = []
	hog = cv2.HOGDescriptor('../hog.xml')
	count = 0

	print("img_nums: ", len(trainset))
	for img in trainset:
		# print("get number",count, " feature")
		count = count+1
		img = np.reshape(img, (28,28))
		cv_img = img.astype(np.uint8)
		hog_feature = hog.compute(cv_img)
		features.append(hog_feature)
	features = np.array(features)
	features = np.reshape(features, (-1, 324))
	return features
def Predict(testset, trainset, train_labels):
	_predict = []
	count = 0

	for test_vec in testset:

		print(count)
		count +=1
		knn_list = []
		max_index = -1
		max_dist = 0

		for i in range(k):
			label = train_labels[i]
			train_vec = trainset[i]

			dist = np.linalg.norm(train_vec - test_vec)

			knn_list.append((dist, label))
		for i in range(k, len(train_labels)):
			label = train_labels[i]
			train_vec = trainset[i]

			dist = np.linalg.norm(train_vec - test_vec)
			if max_index < 0:
				for j in range(k):
					if max_dist < knn_list[j][0]:
						max_index = j
						max_dist = knn_list[max_index][0]
			if dist < max_dist:
				knn_list[max_index] = (dist, label)
				max_index = -1
				max_dist = 0
		class_total = 10
		class_count = [0 for i in range(class_total)]
		for dist, label in knn_list:
			class_count[label] += 1
		mmax = max(class_count)

		for i in range(class_total):
			if mmax == class_count[i]:
				_predict.append(i)
				break
	return np.array(_predict)


if __name__ == '__main__':


	raw_data = pd.read_csv('../data/train.csv', header=0)
	data = raw_data.values
	print("load data success")

	imgs = data[0::,1::]
	labels = data[::,0]

	# compute trian set feature
	print("get feature")
	features = get_hog_feature(imgs)

	print("predict label")
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
	print("train_nums: ", len(train_labels), " test_nums: ", len(test_labels))
	test_predict = Predict(test_features, train_features, train_labels)
	print("predict success")

	print("cal score")
	socre = accuracy_score(test_labels, test_predict)
	print("The accuracy socre is", socre)