from util import *
from ID3_Tree import *
from random import shuffle


def Random_Forest(dataset):
	Forest = list()
	data = dataset
	# create a forest consists of 15 trees
	for i in range(15):
		training_set = data[:30]
		tree = Node(training_set)
		# each tree perform pre-pruning when node contain less than 10 data
		ID3_algorithm(tree)
		Forest.append(tree)
		shuffle(data)

	return Forest

def main():
	file = open('iris.csv')
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
	
	k_fold_time = 1
	test_time = 5
	total = 0
	total_precise = [0.0,0.0,0.0]
	total_recall = [0.0,0.0,0.0]
	total_acc = [0.0,0.0,0.0]
	count = [0,0,0]
	for _ in range(k_fold_time):
		shuffle(data)
		training_set, testing_set = k_fold_dataset(data)
		
		acc = 0
		acc_pre = [0.0,0.0,0.0]
		acc_rec = [0.0,0.0,0.0]
		acc_2 = [0.0,0.0,0.0]
		for i in range(test_time):
			RF = Random_Forest(training_set[i])
			accuracy, precision, recall, total_accuracy, count  = calc_error(testing_set[i],RF)
			for j in range(3):
				acc_pre[j] += precision[j]
				acc_rec[j] += recall[j]
				acc_2[j] += total_accuracy[j]
			acc += accuracy
	
		for k in range(3):
			total_precise[k] += acc_pre[k]/test_time
			total_recall[k] += acc_rec[k]/test_time
			total_acc[k] +=  acc_2[k]/test_time
		total += acc/test_time
	
	true_acc = (total_acc[0]*count[0] + total_acc[1]*count[1] + total_acc[2]*count[2])/(count[0]+count[1]+count[2])
	
	print"{:.3f}".format(total/k_fold_time)
	print"{:.3f} {:.3f} {:.3f}".format(total_precise[0]/k_fold_time,total_recall[0]/k_fold_time,total_acc[0]/k_fold_time)
	print"{:.3f} {:.3f} {:.3f}".format(total_precise[1]/k_fold_time,total_recall[1]/k_fold_time,total_acc[1]/k_fold_time)
	print"{:.3f} {:.3f} {:.3f}".format(total_precise[2]/k_fold_time,total_recall[2]/k_fold_time,total_acc[2]/k_fold_time)


if __name__ == '__main__':
	main()