from Node import Node
from util import *
from math import log
from random import shuffle


def find_impure_leaf(node):
	
	if node is None:
		return None

	if not(node.pure) and node.leaf:
		return node

	left_child = find_impure_leaf(node.left)
	if left_child != None:
		return left_child

	right_child = find_impure_leaf(node.right)
	if right_child != None:
		return right_child

	return None

def find_threshold_spiltIdx(data):
	best_feather_index = -1
	best_entropy = float('inf')
	best_threshold = float('inf')

	for i in range(len(data[0][:-1])):
		(entropy, threshold) = cal_lowest_entropy(data, i)

		if entropy < best_entropy:
			best_feather_index = i
			best_entropy = entropy
			best_threshold = threshold

	return (best_threshold, best_feather_index)

def cal_lowest_entropy(data, feature_index):
	sort_data = sort_by_axis(data, axis = feature_index)
	best_entropy = float('inf')
	best_threshold = float('inf')
	current_entropy = float('inf')
	current_threshold = float('inf')

	for i in range(0, len(data)):
		if i < len(data)-1 :
			current_threshold = (sort_data[i][feature_index] + sort_data[i+1][feature_index])/2
			#print current_threshold

		(left, right) = split(sort_data, current_threshold, feature_index)
		current_entropy = cal_entropy(left) * float(len(left))/float(len(data)) + cal_entropy(right) * float(len(right))/float(len(data))

		if current_entropy < best_entropy:
			best_entropy = current_entropy
			best_threshold = current_threshold

	return (best_entropy, best_threshold)

def cal_entropy(data):
	count = [0,0,0]
	total = float(0)
	# really only need indices 1,2,3 as those are the only labels
	for datapoint in data:
		if datapoint[-1] == 'setosa':
			count[0] += 1
		elif datapoint[-1] == 'versicolor':
			count[1] += 1
		else :
			count[2] += 1
		total = total + 1

	entropy = float(0)
	for c in count:
		if c == 0:
			continue
		prob = c / total
		entropy = entropy - prob * log(prob)
	return entropy

def split(data, threshold, feature_index):
	left = []
	right = []
	for datapoint in data:
		#print datapoint
		if datapoint[feature_index] <= threshold:
			left.append(datapoint)
		else:
			right.append(datapoint)
	return (left,right)

def ID3_algorithm(root):

	current_node = find_impure_leaf(root)

	count = 0
	
	while current_node != None:

		(threshold, feature_index) = find_threshold_spiltIdx(current_node.data)
		(left, right) = split(current_node.data, threshold, feature_index)

		current_node.set_threshold(threshold)
		current_node.set_threshold_idx(feature_index)

		left_node = Node(left)
		right_node = Node(right)
		current_node.left = left_node
		current_node.right = right_node
		current_node.leaf = False

		current_node = find_impure_leaf(root)
		count += 1
	#print "done construct ID3_tree!"

def predict(datapoint, Tree):
	curr_node = Tree
	while not(curr_node.pure):
		threshold = curr_node.threshold
		feature_index = curr_node.threshold_idx
		if datapoint[feature_index] <= threshold:
			curr_node = curr_node.left
		else:
			curr_node = curr_node.right

	return curr_node.label

def calc_error(dataset, Tree):
	errors = 0
	num_samples = len(dataset)
	true_positive = [0.0,0.0,0.0]
	false_positive = [0.0,0.0,0.0]
	false_negative = [0.0,0.0,0.0]
	true_negative = [0.0,0.0,0.0]
	precision = [0.0,0.0,0.0]
	recall = [0.0,0.0,0.0]

	for datapoint in dataset:
		prediction = predict(datapoint, Tree)
		ground_truth = datapoint[-1]

		if not(ground_truth == prediction): # accuracy
			errors = errors + 1

		# need revise
		if prediction == 'setosa':
			if ground_truth == 'setosa':
				true_positive[0] += 1
				true_negative[1] += 1
				true_negative[2] += 1
			elif ground_truth == 'versicolor':
				false_positive[0] += 1
				false_negative[1] += 1
				true_negative[2] += 1
			elif ground_truth == 'virginica':
				false_positive[0] += 1
				true_negative[1] += 1
				false_negative[2] += 1

		elif prediction == 'versicolor':
			if ground_truth == 'setosa':
				false_negative[0] += 1
				false_positive[1] += 1
				true_negative[2] += 1
			elif ground_truth == 'versicolor':
				true_negative[0] += 1
				true_positive[1] += 1
				true_negative[2] += 1
			elif ground_truth == 'virginica':
				true_negative[0] += 1
				false_positive[1] += 1
				false_negative[2] += 1

		elif prediction == 'virginica':
			if ground_truth == 'setosa':
				false_negative[0] += 1
				true_negative[1] += 1
				false_positive[2] += 1
			elif ground_truth == 'versicolor':
				true_negative[0] += 1
				false_negative[1] += 1
				false_positive[2] += 1
			elif ground_truth == 'virginica':
				true_negative[0] += 1
				true_negative[1] += 1
				true_positive[2] += 1



	accuracy = 1 - (float(errors) / float(num_samples))
	for i in range(3):
		precision[i] = float(true_positive[i]) / (float(false_positive[i]) + float(true_positive[i]))
		recall[i] = float(true_positive[i]) / (float(false_negative[i]) + float(true_positive[i]))
		

	return accuracy, precision, recall


def main():
	file = open('./0310120/iris.csv')
	data = []
	# loading iris dataset
	for idx ,line in enumerate(file):
		line = line.strip("\r\n")
		if idx > 0:
			data.append([float(element) for element in line.split(',')[:-1]])
			data[idx].append(line.split(',')[-1])
		else:
			data.append(line.split(','))
	
	# define attributes
	attributes = data[0]
	data.remove(attributes)

	#print data
	std_normalize(data)
	#print "after standard normalize"
	#print data
	
	test_time = 1
	k_fold_time = 5
	total = 0
	total_precise = [0.0,0.0,0.0]
	total_recall = [0.0,0.0,0.0]
	for _ in range(test_time):
		shuffle(data)
		training_set, testing_set = k_fold_dataset(data)
		
		acc = 0
		acc_pre = [0.0,0.0,0.0]
		acc_rec = [0.0,0.0,0.0]
		for i in range(k_fold_time):
			ID3_Tree = Node(training_set[i])
			ID3_algorithm(ID3_Tree)
			accuracy, precision, recall  = calc_error(testing_set[i],ID3_Tree)
			for j in range(3):
				acc_pre[j] += precision[j]
				acc_rec[j] += recall[j]
			acc += accuracy
		
	
		for k in range(3):
			total_precise[k] += acc_pre[k]/k_fold_time
			total_recall[k] += acc_rec[k]/k_fold_time
		total += acc/k_fold_time
	
	#print "\nSummary after ",test_time," times of 5-fold-validation:\n"

	print"{:.3f}".format(total/test_time)
	print"{:.3f} {:.3f}".format(total_precise[0]/test_time,total_recall[0]/test_time)
	print"{:.3f} {:.3f}".format(total_precise[1]/test_time,total_recall[1]/test_time)
	print"{:.3f} {:.3f}".format(total_precise[2]/test_time,total_recall[2]/test_time)	

if __name__ == '__main__':
	main()
