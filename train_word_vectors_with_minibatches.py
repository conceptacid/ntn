import sys
import csv
import random
import tensorflow as tf
import numpy as np
import datetime

use_LBFGS = True
doResetNoWord = False
use_whole_entity_words = False
generate_summaries = False

'''
entities_filename = 'data/Freebase/entities.txt'
relations_filename = 'data/Freebase/relations.txt'
train_data_triplets_filename = 'data/Freebase/train.txt'
test_data_triplets_filename = 'data/Freebase/test.txt'
dev_data_triplets_filename = 'data/Freebase/dev.txt'

words_filename = "freebase_words.txt"

d=100                  # the size of the entity vector
K=4                    # the number of slices in the tensor layer  (K=4)
lambd = 0.0001         # regularization parameter
C = 2                  # number of corrupted examples
NumberOfThresholdSteps = 100

'''

entities_filename = 'data/Wordnet/entities.txt'
relations_filename = 'data/Wordnet/relations.txt'
train_data_triplets_filename = 'data/Wordnet/train.txt'
test_data_triplets_filename = 'data/Wordnet/test.txt'
dev_data_triplets_filename = 'data/Wordnet/dev.txt'

if use_whole_entity_words:
    words_filename = "data/Wordnet/entities.txt"
else:
    words_filename = "wordnet_words.txt"

d = 100  # the size of the entity vector
K = 4  # the number of slices in the tensor layer  (K=4)
lambd = 0.5  # regularization parameter
C = 10  # number of corrupted examples
NumberOfThresholdSteps = 100
num_batches = 50

max_epochs = 10

time_start = datetime.datetime.now()
print("Starting new session ", time_start.isoformat())

if len(sys.argv) > 1:
    d = int(sys.argv[1])
    K = int(sys.argv[2])
    num_batches = int(sys.argv[3])
    C = int(sys.argv[4])
    lambd = float(sys.argv[5])
    max_epochs = float(sys.argv[6])

print(" d=",d)
print(" K=",K)
print(" num_batches=",num_batches)
print(" C=",C)
print(" lambd=",lambd)
print(" max_epochs=", max_epochs)
print(" max_iterations=", max_epochs*num_batches)


#exit(77)

def read_words(filename):
    res = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            res.append(row[0])
    return res


def read_ids(filename):
    res = {}
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            res[row[0]] = counter
            counter += 1
    return res


def read_tuples(filename, entity_lookup={}, relation_lookup={}, read_groud_truth=False):
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
            quad = triplet + (random.randint(0, entity_size - 1),)
            result.append(quad)
    return result


def format_dev_test_data(data):
    result = np.zeros((int(len(data) / 2), 4))
    for i in range(0, len(data), 2):
        assert (data[i][0] == data[i + 1][0])
        assert (data[i][1] == data[i + 1][1])
        assert (data[i][3] == 1)
        assert (data[i + 1][3] == -1)
        j = (int)(i / 2)
        result[j, 0] = data[i][0]  # E1
        result[j, 1] = data[i][1]  # R
        result[j, 2] = data[i][2]  # E2
        result[j, 3] = data[i + 1][2]  # C
    return result


def create_whole_entity_words(entity_lookup, words):
    return np.array(range(len(entity_lookup))).reshape(1, -1), 1


# entity_lookup is a dict from entity [w,w,w,...] to index of that entity
# create a lookup dict word -> index
def create_entity_words(entity_lookup, words):
    word_to_index = dict([(key, val) for val, key in enumerate(words)])

    # for each entity add the list of words of that entity
    entity_word_strings = [None] * len(entity_lookup)
    num_words_per_entity = 0
    for entity_str, index in entity_lookup.items():
        words = list(filter(lambda w: len(w) > 0, entity_str.split("_")))
        # print(words[:-1])
        words = words[:-1]  # only valid for wordnet
        entity_word_strings[index] = words
        num_words_per_entity = max(num_words_per_entity, len(words))

    # for each entity lookup the indices of the words
    entity_word_indices = np.zeros(shape=(num_words_per_entity, len(entity_word_strings)), dtype=np.int64)
    for entity_index, word_strings in enumerate(entity_word_strings):
        for i, word in enumerate(word_strings):
            entity_word_indices[i, entity_index] = word_to_index[word]

    # remove stop words


    return entity_word_indices, num_words_per_entity


def define_parameters(d=d, Ns=None, Nr=None, K=K):
    params = dict(
        S=tf.get_variable(shape=(Ns, d), name="S", initializer=tf.contrib.layers.xavier_initializer(),
                          regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
    )

    for r in range(Nr):
        U = tf.get_variable(shape=(K, 1), name="U_" + str(r), initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
        V = tf.get_variable(shape=(K, 2 * d), name="V_" + str(r), initializer=tf.contrib.layers.xavier_initializer(),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
        b = tf.get_variable(shape=(K, 1), name="b_" + str(r))
        # ,   initializer=tf.zeros_initializer(),
        # regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
        params["U_" + str(r)] = U
        params["V_" + str(r)] = V
        params["b_" + str(r)] = b
        for k in range(K):
            name = "W_" + str(r) + "_" + str(k)
            params[name] = tf.get_variable(shape=(d, d), name=name, initializer=tf.contrib.layers.xavier_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))

        threshold_min = 0.0
        threshold_max = 10.0
        threshold_steps = NumberOfThresholdSteps
        range_of_thresholds = np.linspace(threshold_min, threshold_max, num=threshold_steps, dtype=np.float32)
        range_of_thresholds = np.reshape(range_of_thresholds, (-1, 1))

        th = tf.Variable(range_of_thresholds, dtype=tf.float32, name="range_of_thresholds")
        params["th_" + str(r)] = th

    return params


def build_g(params, E1, E2, r):
    SlicesArray = []
    for k in range(K):
        Wrk = params["W_" + str(r) + "_" + str(k)]

        # print("Wrk.shape=", Wrk.shape)
        # print("E2.shape=", E2.shape)

        Temp = tf.multiply(E1, tf.matmul(Wrk, E2))
        Slice = tf.reduce_sum(Temp, axis=0, keep_dims=True)

        # print("Slice.shape = ", Slice.shape)

        SlicesArray.append(Slice)
    eWe = tf.concat(SlicesArray, axis=0)

    # Vr should have shape (K,2*d)
    Vr = params["V_" + str(r)]
    assert (Vr.shape == (K, 2 * d))

    # br should have shape (K,1)
    br = params["b_" + str(r)]
    assert (br.shape == (K, 1))

    # print("EWE.shape =", eWe.shape)

    Activation = tf.tanh(eWe + tf.matmul(Vr, tf.concat([E1, E2], axis=0)) + br)

    g = tf.matmul(tf.transpose(params["U_" + str(r)]), Activation)
    assert (g.shape[0] == 1)
    return g


# Extracts training data entities from indices in X
def build_entity_lookup(params, X, EW, r, data_row, name_prefix, num_words_per_entity):
    E_indices = tf.slice(X, begin=(data_row, 0), size=(1, -1), name=name_prefix + "_index_" + str(r))
    #print("shape of E_indices of", name_prefix, E_indices.shape)  # (1, number_of_samples)

    S_indices = tf.gather(EW, E_indices, axis=1, name="S_indices_" + name_prefix + "_" + str(r))
    #print("shape of S_indices1", S_indices.shape)

    # S_indices = tf.reshape(S_indices,(num_words_per_entity,-1))
    S_indices = tf.squeeze(S_indices, axis=1)
    #print("shape of S_indices2", S_indices.shape)  # (num_words_per_entity, number_of_samples)

    WordVectors = tf.gather(tf.transpose(params["S"]), S_indices, axis=1,
                            name="WordVectors_" + name_prefix + "_" + str(r))
    WordVectors = tf.transpose(WordVectors, perm=(0, 2, 1))
    # WordVectors = tf.reshape(WordVectors,(d,-1, num_words_per_entity))  # shape=(d, number_of_samples, num_words_per_entity)
    #print("shape of WordVectors", WordVectors.shape)

    if use_whole_entity_words:
        E = tf.reduce_sum(WordVectors, axis=2, keep_dims=False, name=name_prefix + "_" + str(r))  # E1, E2, C
        #print("shape of", name_prefix, E.shape)

        count = tf.constant(1.0)  # , shape=(1, S_indices.shape[1]))

        # tf.count_nonzero( S_indices, axis=0, keep_dims=False, dtype=tf.float32)

        # condition =  (tf.sign( tf.cast(S_indices, tf.float32) - tf.constant(0.5) ) + tf.constant(1.0))/tf.constant(2.0)
        # count = tf.reduce_sum(condition, axis=0, keep_dims=True)

        # tf.reduce_sum(tf.cast( S_indices > 0, tf.float32 ), axis=0, keep_dims=True)
        #print("shape of count", count.shape)
        E = tf.div(E, count)
        #print("shape of", name_prefix, E.shape)

        return E
        # return tf.squeeze(WordVectors)
    else:
        E = tf.reduce_sum(WordVectors, axis=2, keep_dims=False, name=name_prefix + "_" + str(r))  # E1, E2, C
        #print("shape of", name_prefix, E.shape)
        # count = tf.cast( tf.reduce_sum(tf.cast( S_indices > 0, tf.int64 ), axis=0, keep_dims=True), tf.float32 )
        # print("shape of count", count.shape)
        # E = tf.div(E,count)
        # print("shape of", name_prefix, E.shape)
        return E  # E shape should be (d,m_r)


def define_graph(params, Nr, K, Ns, num_words_per_entity, Ne):
    Xs = {}
    sums_r = []
    hits = []

    global_true_positives = []
    global_true_negatives = []
    global_false_positives = []
    global_false_negatives = []

    # Accuracy calculation:
    # we want to do a linear search over the range of thresholds
    # we have to define a column for a range of thresholds
    # each value will be used in combination with each g (row vector) using broadcasting to compute different value of accuracy per threshold
    # step by step example:
    # 1) we get a row vector of g (also gc) - one column per sample
    # 2) then we compare each of them wigh each threshold, this gives us the matrix of shape (num thresholds x num samples)
    # 3) then we compute true positives, true negatives, etc.. for each row
    # 4) then we calculate accuracy for each row
    # 5) thtn we find the max accuracy and that gives us also the optimal threshold in the grid
    # th_values

    # todo: we build range of thresholds for each relation
    # for now we start with one hardcoded range


    EW = tf.placeholder(dtype=tf.int64, shape=(num_words_per_entity, Ne), name="EntityWords")

    for r in range(Nr):
        rannge_of_thresholds = params["th_" + str(r)]

        X = tf.placeholder(dtype=tf.int64, shape=(4, None), name="X_" + str(r))

        # group entity indices by relation index r
        Xs["X_" + str(r)] = X

        E1 = build_entity_lookup(params, X, EW, r, 0, "E1", num_words_per_entity)
        E2 = build_entity_lookup(params, X, EW, r, 2, "E2", num_words_per_entity)
        C = build_entity_lookup(params, X, EW, r, 3, "C", num_words_per_entity)

        gc = build_g(params, E1, C, r)
        bound1 = tf.reduce_mean(gc)

        g = build_g(params, E1, E2, r)
        bound2 = tf.reduce_mean(g)

        delta = (bound2 - bound1) / (NumberOfThresholdSteps - 1)
        new_range = tf.reshape(tf.range(start=bound1, limit=bound2, delta=delta, dtype=tf.float32),
                               rannge_of_thresholds.shape)
        rannge_of_thresholds.assign(new_range)

        # compute best threshold and then accuracy for this relation

        true_positives = tf.reduce_sum(tf.cast(g > rannge_of_thresholds, tf.float32), axis=1, keep_dims=True)
        false_negatives = tf.reduce_sum(tf.cast(g <= rannge_of_thresholds, tf.float32), axis=1, keep_dims=True)

        false_positives = tf.reduce_sum(tf.cast(gc > rannge_of_thresholds, tf.float32), axis=1, keep_dims=True)
        true_negatives = tf.reduce_sum(tf.cast(gc <= rannge_of_thresholds, tf.float32), axis=1, keep_dims=True)

        # this is a column vector per threshold
        accuracy_of_relation = (true_positives + true_negatives) / (
        true_positives + true_negatives + false_positives + false_negatives)

        # we pick the maximum accuracy of the relation
        best_threshold_index = tf.squeeze(tf.argmax(accuracy_of_relation, axis=0))

        # we get the best_threshold as a number
        best_threshold = rannge_of_thresholds[best_threshold_index, 0]

        global_true_positives.append(true_positives[best_threshold_index, 0])
        global_false_negatives.append(false_negatives[best_threshold_index, 0])

        global_false_positives.append(false_positives[best_threshold_index, 0])
        global_true_negatives.append(true_negatives[best_threshold_index, 0])

        sum_r = tf.reduce_sum(tf.maximum(tf.constant(0.), tf.constant(1.) - g + gc))
        sums_r.append(sum_r)

    sum_true_positives = tf.add_n(global_true_positives)
    sum_false_positives = tf.add_n(global_false_positives)
    sum_true_negatives = tf.add_n(global_true_negatives)
    sum_false_negatives = tf.add_n(global_false_negatives)

    accuracy = (sum_true_positives + sum_true_negatives) / (
    sum_true_positives + sum_true_negatives + sum_false_positives + sum_false_negatives)

    regularization_cost = 100.0 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = tf.add_n(sums_r) + regularization_cost
    tf.summary.scalar('cost', cost)

    # L-BFGS optimizer
    if use_LBFGS:
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={
            'maxiter': 1})  # if 'method' arg is undefined, the default method is L-BFGS-B
    else:
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

    S = params["S"]

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('regularization_cost', regularization_cost)


    resetNoword = tf.scatter_update(S, indices=[0], updates=tf.zeros(shape=[1, d]))

    return EW, Xs, cost, optimizer, accuracy, regularization_cost, resetNoword


def create_data_feed(Xs, quadruplets, Nr, E, entity_words):
    res = {}
    res[E] = entity_words
    for r in range(Nr):
        name = "X_" + str(r)
        res[Xs[name]] = quadruplets[:, quadruplets[1, :] == r]
        # print(" data feed X_",r, " is ", res[Xs[name]].shape, res[Xs[name]].dtype )
        # print("r=",r, "\n", quadruplets[1, :] == r)
    return res





words = ["NOWORD"] + read_words(words_filename)

entity_lookup = read_ids(entities_filename)
relation_lookup = read_ids(relations_filename)

Ns = len(words)
Ne = len(entity_lookup)
Nr = len(relation_lookup)
print("Number of words = ", Ns)
print("Number of entities = ", Ne)
print("Number of relations = ", Nr)

train_data_tuples = read_tuples(train_data_triplets_filename, entity_lookup=entity_lookup,
                                relation_lookup=relation_lookup)

# this data looks "ID-like maria_anna_of_sardinia   ID-gender  ID-female  1"
test_data_tuples = read_tuples(test_data_triplets_filename, entity_lookup=entity_lookup,
                               relation_lookup=relation_lookup, read_groud_truth=True)
dev_data_tuples = read_tuples(dev_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup,
                              read_groud_truth=True)

if use_whole_entity_words:
    entity_words, num_words_per_entity = create_whole_entity_words(entity_lookup, words)
else:
    entity_words, num_words_per_entity = create_entity_words(entity_lookup, words)

print("num_words_per_entity = ", num_words_per_entity)

train_data = np.array(add_corrupted_exampes(train_data_tuples, C, Ne)).T
dev_data = np.array(format_dev_test_data(dev_data_tuples)).T

# print(add_corrupted_exampes(train_data, 2, len(entity_lookup)))
params = define_parameters(Ns=Ns, Nr=len(relation_lookup))
E, Xs, cost, optimizer, accuracy, regularization_cost, resetNoword = define_graph(params, Nr, K, Ns,
                                                                                  num_words_per_entity,
                                                                                  Ne)  # params, Nr, K, Ns, num_words_per_entity, Ne




train_data_feed = []


batch_size = int(train_data.shape[1]/num_batches)

for i in range(num_batches):
    batch_end = (i+1)*batch_size
    if i == num_batches-1:
        batch_end = train_data.shape[1]
    train_data_slice = train_data[:,i*batch_size:batch_end]
    train_data_feed.append( create_data_feed(Xs, train_data_slice, Nr, E, entity_words) )
    print("train data batch shape=", train_data_slice.shape)

dev_data_feed = create_data_feed(Xs, dev_data, Nr, E, entity_words)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summary/train', sess.graph)
    test_writer = tf.summary.FileWriter('summary/test')
    sess.run(init)

    train_data_index = 0
    epoch_num = 0
    i = 0
    while epoch_num < max_epochs:
        train_data_batch_feed = train_data_feed[train_data_index]
        if doResetNoWord:
            sess.run(resetNoword)
        summary, cost_value, accuracy_value, regularization_cost_value = sess.run([merged, cost, accuracy, regularization_cost], feed_dict=train_data_batch_feed)

        if i == 0:
            print("\n\n{0:10} {1:10} {2:10} {3:10} {4:10} {5:10}".format(
                "  num_iter",
                "train_cost", "train_accu",
                " dev_cost", "  dev_accu", "regulariza"))

        if train_data_index==(num_batches-1):
            epoch_num += 1
            dev_summary, dev_cost_value, dev_accuracy_value = sess.run([merged, cost, accuracy], feed_dict=dev_data_feed)
            if generate_summaries:
                test_writer.add_summary(dev_summary, i)
            print("{0:10} {1:10.4} {2:10.4} {3:10.4} {4:10.4} {5:10.4}".format(
                i,
                cost_value, accuracy_value,
                dev_cost_value, dev_accuracy_value,
                regularization_cost_value))

        print("{0:10} {1:10.4} {2:10.4} {3:10.4} {4:10.4} {5:10.4}".format(
            i,
            cost_value, accuracy_value,
            " ", " ",
            regularization_cost_value))

        if use_LBFGS:
            optimizer.minimize(sess, feed_dict=train_data_batch_feed)
        else:
            sess.run(optimizer, feed_dict=train_data_batch_feed)

        train_data_index = (train_data_index+1)%num_batches

        if generate_summaries:
            train_writer.add_summary(summary, i)
        i += 1

    dev_accuracy_value = sess.run([accuracy], feed_dict=dev_data_feed)
    print("final_accuracy="+dev_accuracy_value[0])

time_end = datetime.datetime.now()
print("Ending session ", time_end.isoformat())
print("Elapsed time ", (time_end - time_start).total_seconds())