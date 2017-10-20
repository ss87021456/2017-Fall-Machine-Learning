import numpy as np

def k_fold_dataset(data):
	training_1 = data[:int(len(data)*0.8)] 	# 0% ~ 80%
	test_1 = data[int(len(data)*0.8):] 		# 80% ~ 100%
	training_2 = data[:int(len(data)*0.6)] + data[int(len(data)*0.8):]	# 0% ~ 60% + 80% ~ 100%
	test_2 = data[int(len(data)*0.6):int(len(data)*0.8)]				# 60% ~ 80%
	training_3 = data[:int(len(data)*0.4)] + data[int(len(data)*0.6):]	# 0% ~ 40% + 60% ~ 100%
	test_3 = data[int(len(data)*0.4):int(len(data)*0.6)]				# 40% ~ 60%
	training_4 = data[:int(len(data)*0.2)] + data[int(len(data)*0.4):]	# 0% ~ 20% + 40% ~ 100%
	test_4 = data[int(len(data)*0.2):int(len(data)*0.4)]				# 20% ~ 40%
	training_5 = data[int(len(data)*0.2):]								# 20% ~ 100%
	test_5 = data[:int(len(data)*0.2)]									# 0% ~ 20%
	
	training_set = []
	training_set.append(training_1)
	training_set.append(training_2)
	training_set.append(training_3)
	training_set.append(training_4)
	training_set.append(training_5)
	testing_set = []
	testing_set.append(test_1)
	testing_set.append(test_2)
	testing_set.append(test_3)
	testing_set.append(test_4)
	testing_set.append(test_5)

	return (training_set, testing_set)

def sort_by_axis(data, axis):

	data = np.asarray(data)
	# apply lex sort to sort 2-D data with specific colomn
	ind = np.lexsort((data[:,axis],data[:,axis]))
	data = data[ind]
	new_data = []
	# need to convert data type from str back to float
	for idx, element in enumerate(data):
		new_data.append([float(item) for item in element[:-1]]) #avoid change last element type (str)
		new_data[idx].append(element[-1])						#conpensate the last class

	return new_data