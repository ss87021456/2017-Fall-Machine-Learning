from Node_rf import Node
from util import *
from math import log

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
	#print len(data[0][:-1])

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

def predict(datapoint, Forest):
	count = [0,0,0]
	prediction = None
	#print len(Forest)
	for tree in Forest:
		curr_node = tree
		while not(curr_node.pure):
			threshold = curr_node.threshold
			feature_index = curr_node.threshold_idx
			if datapoint[feature_index] <= threshold:
				curr_node = curr_node.left
			else:
				curr_node = curr_node.right
		if curr_node.label == 'setosa':
			count[0] += 1
		elif curr_node.label == 'versicolor':
			count[1] += 1
		elif curr_node.label == 'virginica':
			count[2] += 1

	max_idx = max(enumerate(count),key=lambda x: x[1])[0]

	if max_idx == 0:
		#print 'setosa'
		prediction = 'setosa'
	elif max_idx == 1:
		#print 'versicolor'
		prediction = 'versicolor'
	else:
		#print 'virginica'
		prediction = 'virginica'

	return prediction

def calc_error(dataset, Forest):
	errors = 0
	num_samples = len(dataset)
	true_positive = [0.0,0.0,0.0]
	false_positive = [0.0,0.0,0.0]
	false_negative = [0.0,0.0,0.0]
	true_negative = [0.0,0.0,0.0]
	precision = [0.0,0.0,0.0]
	recall = [0.0,0.0,0.0]
	total_accuracy = [0.0,0.0,0.0]
	count = [0,0,0]
	for datapoint in dataset:
		prediction = predict(datapoint, Forest)
		ground_truth = datapoint[-1]

		if not(ground_truth == prediction): # accuracy
			errors = errors + 1

		# need revise
		if prediction == 'setosa':
			if ground_truth == 'setosa':
				true_positive[0] += 1
				true_negative[1] += 1
				true_negative[2] += 1
				count[0] += 1
			elif ground_truth == 'versicolor':
				false_positive[0] += 1
				false_negative[1] += 1
				true_negative[2] += 1
				count[1] += 1
			elif ground_truth == 'virginica':
				false_positive[0] += 1
				true_negative[1] += 1
				false_negative[2] += 1
				count[2] += 1

		elif prediction == 'versicolor':
			if ground_truth == 'setosa':
				false_negative[0] += 1
				false_positive[1] += 1
				true_negative[2] += 1
				count[0] += 1
			elif ground_truth == 'versicolor':
				true_negative[0] += 1
				true_positive[1] += 1
				true_negative[2] += 1
				count[1] += 1
			elif ground_truth == 'virginica':
				true_negative[0] += 1
				false_positive[1] += 1
				false_negative[2] += 1
				count[2] += 1

		elif prediction == 'virginica':
			if ground_truth == 'setosa':
				false_negative[0] += 1
				true_negative[1] += 1
				false_positive[2] += 1
				count[0] += 1
			elif ground_truth == 'versicolor':
				true_negative[0] += 1
				false_negative[1] += 1
				false_positive[2] += 1
				count[1] += 1
			elif ground_truth == 'virginica':
				true_negative[0] += 1
				true_negative[1] += 1
				true_positive[2] += 1
				count[2] += 1


	accuracy = 1 - (float(errors) / float(num_samples))
	for i in range(3):
		precision[i] = float(true_positive[i]) / (float(false_positive[i]) + float(true_positive[i]))
		recall[i] = float(true_positive[i]) / (float(false_negative[i]) + float(true_positive[i]))
		total_accuracy[i] = (float(true_negative[i]) + float(true_positive[i]))/ (float(true_negative[i]) + float(true_positive[i]) + float(false_negative[i]) + float(false_positive[i]))

	return accuracy, precision, recall, total_accuracy, count