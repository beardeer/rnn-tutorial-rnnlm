#! /usr/bin/env python

import numpy as np

def dkt_input_data_reader(input_file_name = './data/dkt_input_2.txt'):
	input_file = open(input_file_name, 'r')
	train_data = []
	test_data = []
	for i in input_file.readlines():
		data = i.strip().split(',')
		data = map(int, data)
		data = np.asarray(data)
		data = data + 2
		data = data.tolist()

		if len(data) < 2:
			pass
		train = [0] + data[:-1]
		test = data[1:] + [1]

		if len(train) != len(test):
			print 'input data error!'
			print train
			print test
			raise ValueError

		train_data.append(train)
		test_data.append(test)

	return train_data, test_data

if __name__ == "__main__": 
	a,b = dkt_input_data_reader()

	print len(a)
	print len(b)