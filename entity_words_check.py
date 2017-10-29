import tensorflow as tf
import numpy as np


entities = ["tokyo", "university of tokyo", "university of zagreb", "zagreb"]
words = ["NOWORD", "tokyo", "university", "of", "zagreb"]

num_words_per_entity = 3
Ns = len(words)
Ne = len(entities)
d = 6 # number of dimensions of word vectors

EW = tf.constant([ [1, 2, 2, 4],
                   [0, 3, 3, 0],
                   [0, 1, 4, 0] ], dtype=tf.int64)

S = tf.constant([ [0, 1, 2, 3, 4],
                  [0, 1, 2, 3, 4],
                  [0, 1, 2, 3, 4],
                  [0, 1, 2, 3, 4],
                  [0, 1, 2, 3, 4],
                  [0, 1, 2, 3, 4] ], dtype=tf.float64)

E_indices = tf.constant([ [1, 0] ], dtype=tf.int64)

S_indices = tf.gather( EW, E_indices, axis=1, name="S_indices")
S_indices = tf.reshape(S_indices,(num_words_per_entity,-1))	

WordVectors = tf.transpose(tf.gather( S, S_indices, axis=1, name="WordVectors"), perm=(0,2,1))
print("shape of WordVectors", WordVectors.shape)


E = tf.reduce_mean(WordVectors, axis=2, keep_dims=False, name="E")

with tf.Session() as sess:
	val = sess.run(E)		
	print(val)