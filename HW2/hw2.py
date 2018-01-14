import pandas as pd
import numpy as np
import sys
from kdtree import *

# HYPERPARAMETER
valid_num = 36
k_list = [1,5,10,100]
dim = 9
target = ['cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS']

def prediction(index, labels):
    global target
#cp, im, pp, imU, om, omL, inL, imS 8 classes
    count = [0 for i in range(len(target))]
    votes = [labels[idx] for idx in index]
    for vote in votes:
        for idx, element in enumerate(target):
            if element == vote:
                count[idx] += 1

    result_index = count.index(max(count))

    return target[result_index]

#f = open("output.txt",'w')

#print "Loading data..."
data_path = sys.argv[1]
test_path = sys.argv[2]
df = pd.read_csv(data_path, sep=',')
df_test = pd.read_csv(test_path, sep=',')

# seperate labels, index and features
labels = df_test['10']
train_features = df.drop(['index','0','10'],axis=1)
test_features = df_test.drop(['index','0','10'],axis=1)

# transfer from dataframe to numpy array and then transfer to list
train_features = train_features.as_matrix()
train_features = train_features.tolist()
temp_list = list(train_features)
test_features = test_features.as_matrix()
test_features = test_features.tolist()

#print train_features.index(train_features[0])
#print "Creating k-d tree...\n"
kdtree = create_kd_tree(temp_list, dim)

for k in k_list:
	correct = 0.0
	for test in test_features:
		index = []
		neighbors = naive_knn(kdtree, test, k, dim, lambda a,b: sum((a[i] - b[i]) ** 2 for i in xrange(dim)))

		for element in neighbors:
			index.append(train_features.index(element[1]))
	
		result = prediction(index,labels)
	
		if result == labels[test_features.index(test)]:
			correct += 1

	text = "KNN accuracy: "+str(round(correct/len(test_features),5))
	print text
	#f.write(text+"\n")
	for test in test_features[-3:]:
		neighbors = naive_knn(kdtree, test, k, dim, lambda a,b: sum((a[i] - b[i]) ** 2 for i in xrange(dim)))
			
		for element in neighbors:
			#f.write(str(train_features.index(element[1]))+" ")
			print train_features.index(element[1]),
		#f.write("\n")
		print 
	#f.write("\n")
	print 
### START PCA ###
dim = 7
features = np.asarray(train_features)
mean = np.mean(features, axis=0)
data_matrix = features.copy()
data_matrix = np.subtract(data_matrix, mean)

covariance_matrix = np.dot(data_matrix.T, data_matrix)
w, v = np.linalg.eigh(covariance_matrix)
projection = np.dot(features, np.array([v[:,-1],v[:,-2],v[:,-3],v[:,-4], 
										v[:,-5],v[:,-6],v[:,-7]]).T)
projection = projection.astype(np.float16).tolist()
#print "Creating k-d tree for PCA...\n"
kdtree_pca = create_kd_tree(projection, dim)

temp = np.dot(features,  np.array([v[:,-1],v[:,-2],v[:,-3],v[:,-4],v[:,-5],v[:,-6],v[:,-7]]).T)
temp = temp.astype(np.float16).tolist()
#temp = temp.tolist()

features = features.tolist()

test_features = np.asarray(test_features)

correct = 0.0
for test in test_features :
	test_pca = np.dot(test,np.array([v[:,-1],v[:,-2],v[:,-3],v[:,-4],v[:,-5],v[:,-6],v[:,-7]]).T)
	test_pca = test_pca.astype(np.float16).tolist()
	index = []
	neighbors = naive_knn(kdtree_pca, test_pca, 5, dim, lambda a,b: sum((a[i] - b[i]) ** 2 for i in xrange(dim)))

	for element in neighbors:
		index.append(temp.index(element[1]))

	result = prediction(index,labels)

	if result == labels[features.index(test.tolist())]:
		correct += 1
### END PCA ###
text = "K = 5, KNN_PCA accuracy: "+str(round(correct/len(test_features),5))
print text
#f.write(text)
