import csv
import random
import tensorflow as tf
import numpy as np

entities_filename = 'data/Freebase/entities.txt'
relations_filename = 'data/Freebase/relations.txt'
train_data_triplets_filename = 'data/Freebase/train.txt'
test_data_triplets_filename = 'data/Freebase/test.txt'
dev_data_triplets_filename = 'data/Freebase/dev.txt'

d=100
K=4
lambd = 0.0001

def read_ids(filename):
	res = {}
	counter = 0
	with open(filename, 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			res[row[0]] = counter
			counter +=1 
	return res

def read_triplets(filename, entity_lookup={}, relation_lookup={}, read_groud_truth=False):
	res = []
	with open(filename, 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			if read_groud_truth:
				e1, r, e2, ground_truth = row
				res.append((entity_lookup[e1], relation_lookup[r], entity_lookup[e2], int(ground_truth)))
			else:
				e1, r, e2 = row
				res.append((entity_lookup[e1], relation_lookup[r], entity_lookup[e2]))
	return res

def add_corrupted_exampes(train_data, C, entity_size):
	result = []
	for triplet in train_data:
		for c in range(C):
			quad = triplet + (random.randint(0, entity_size-1),)
			result.append(quad)
	return result

def define_parameters(d=d, Ne=None, Nr=None, K=K):
	params = dict(
		U=tf.get_variable(shape=(K,1), name="U",   initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd)),
		V=tf.get_variable(shape=(K,2*d), name="V", initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd)),
		b=tf.get_variable(shape=(K,1), name="b",   initializer=tf.zeros_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd)),
		E=tf.get_variable(shape=(d,Ne), name="E",  initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
	)
	for r in range(Nr):
		for k in range(K):
			name = "W_"+ str(r) + "_"+str(k)
			params[name] = tf.get_variable(shape=(d,d), name=name, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
	return params

def define_graph(params, batch_size):
	X=tf.placeholder(dtype=tf.float64, shape=(4,batch_size))

	e1_indices = tf.slice(X, begin=(0,0), size=(1,batch_size) )
	e1 = tf.gather( params["E"], e1_indices, axis=1 )

	r_indices = tf.slice(X, begin=(1,0), size=(1,batch_size) )

	e2_indices = tf.slice(X, begin=(2,0), size=(1,batch_size) )
	e2 = tf.gather( params["E"], e2_indices, axis=1 )

	c_indices = tf.slice(X, begin=(3,0), size=(1,batch_size) )
	c = tf.gather( params["E"], c_indices, axis=1 )


	#for k in range(K):
		#Wr = params["W_" + str(r)] ...

	return X, g, cost



entity_lookup = read_ids(entities_filename)
relation_lookup = read_ids(relations_filename)

Ne = len(entity_lookup)
Nr = len(relation_lookup)

train_data = read_triplets(train_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup)
test_data = read_triplets(test_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup, read_groud_truth=True)
dev_data = read_triplets(dev_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup, read_groud_truth=True)


#print(add_corrupted_exampes(train_data, 2, len(entity_lookup)))
p = define_parameters(Ne=Ne, Nr=len(relation_lookup))
print(p)

