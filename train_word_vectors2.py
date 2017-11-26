import csv
import random
import tensorflow as tf
import numpy as np

'''
entities_filename = 'data/Freebase/entities.txt'
relations_filename = 'data/Freebase/relations.txt'
train_data_triplets_filename = 'data/Freebase/train.txt'
test_data_triplets_filename = 'data/Freebase/test.txt'
dev_data_triplets_filename = 'data/Freebase/dev.txt'

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
words_filename = "wordnet_words.txt"

d = 100  # the size of the entity vector
K = 4  # the number of slices in the tensor layer  (K=4)
lambd = 0.5  # regularization parameter
C = 4  # number of corrupted examples
NumberOfThresholdSteps = 100


def read_words(filename):
    res = []
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            res.append(row[0])
    return res


# returns a dict lookup from strings to indices
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


def create_entity_words(entity_to_index_lookup, words):
    word_to_index_lookup = dict([(key, val) for val, key in enumerate(words)])

    # for each entity add the list of words of that entity
    list_of_entity_words = [None] * len(entity_to_index_lookup)
    for entity_str, index in entity_to_index_lookup.items():
        words = list(filter(lambda w: len(w) > 0, entity_str.split("_")))
        # print(words[:-1])
        words = words[:-1]  # only valid for wordnet
        list_of_entity_words[index] = words

    return list_of_entity_words, word_to_index_lookup


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


def define_parameters(d, Nw, Nr, K):
    assert d > 0
    assert Nw > 0
    assert Nr > 0
    assert K > 0

    params = dict(
        WV=tf.get_variable(shape=(d, Nw), name="WV", initializer=tf.contrib.layers.xavier_initializer(),
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
def build_entity_lookup(params, X, E, r, data_row, name_prefix):
    E_indices = tf.slice(X, begin=(data_row, 0), size=(1, -1), name=name_prefix + "_index_" + str(r))
    E_slice = tf.gather(E, E_indices, axis=1, name=name_prefix + "_" + str(r))
    E_slice = tf.reshape(E_slice, (d, -1))
    return E_slice  # E_slice shape should be (d,m_r)


def define_graph(params, Nr, K, word_to_index_lookup, list_of_entity_words):
    assert Nr > 0
    assert K > 0


    #print(word_to_index_lookup)

    Xs = {}
    sums_r = []
    hits = []

    WV = params["WV"]

    global_true_positives = []
    global_true_negatives = []
    global_false_positives = []
    global_false_negatives = []


    print("preparing list of entity word indices...")

    list_of_entity_word_indices = []  # list of constant tensors
    for words_of_entity in list_of_entity_words:
        word_indices_of_entity = []
        for word in words_of_entity:
            word_indices_of_entity.append(word_to_index_lookup[word])
        list_of_entity_word_indices.append(tf.constant(np.array(word_indices_of_entity), dtype=tf.int64))

    print("done with ", len(list_of_entity_word_indices), "items")

    print("preparing list of entity word vectors...")
    list_of_entity_word_vectors = []  # each entity has different number of word vectors, we put them into a list
    for word_indices_of_entity in list_of_entity_word_indices:
        # axis=1 because the shape of WV is (d, Nw), therefore we have to pick a column for each index
        list_of_entity_word_vectors.append(tf.gather(WV, indices=word_indices_of_entity, axis=1))
    print("done with ", len(list_of_entity_word_vectors), "items")

    print("preparing list of entity vectors...")
    list_of_entity_vectors = []
    for entity_word_vectors in list_of_entity_word_vectors:
        # entity_word_vectors is a tensor of shape (d, number_of_words_in_entity)
        list_of_entity_vectors.append(tf.reduce_mean(entity_word_vectors, keep_dims=True, axis=1))
    print("done with ", len(list_of_entity_vectors), "items")

    print("stacking entity vectors...")
    # create a tensor from all these entity vectors in the list
    E = tf.stack(list_of_entity_vectors, axis=1)

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


    print("building the rest of the graph...")
    for r in range(Nr):

        print("  - relation ", r)

        rannge_of_thresholds = params["th_" + str(r)]

        X = tf.placeholder(dtype=tf.int64, shape=(4, None), name="X_" + str(r))

        # group entity indices by relation index r
        Xs["X_" + str(r)] = X

        E1 = build_entity_lookup(params, X, E, r, 0, "E1")
        E2 = build_entity_lookup(params, X, E, r, 2, "E2")
        C = build_entity_lookup(params, X, E, r, 3, "C")

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

    print("building the accuracy, cost graph...")

    sum_true_positives = tf.add_n(global_true_positives)
    sum_false_positives = tf.add_n(global_false_positives)
    sum_true_negatives = tf.add_n(global_true_negatives)
    sum_false_negatives = tf.add_n(global_false_negatives)

    accuracy = (sum_true_positives + sum_true_negatives) / (
    sum_true_positives + sum_true_negatives + sum_false_positives + sum_false_negatives)

    regularization_cost = 100.0 * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    cost = tf.add_n(sums_r) + regularization_cost
    tf.summary.scalar('cost', cost)

    print("creating optimizer...")

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={
        'maxiter': 1})  # if 'method' arg is undefined, the default method is L-BFGS-B

    print("done building graph")

    return Xs, cost, optimizer, accuracy, regularization_cost


def create_data_feed(Xs, quadruplets, Nr):
    res = {}
    for r in range(Nr):
        name = "X_" + str(r)
        res[Xs[name]] = quadruplets[:, quadruplets[1, :] == r]
        # print(" data feed X_",r, " is ", res[Xs[name]].shape, res[Xs[name]].dtype )
        # print("r=",r, "\n", quadruplets[1, :] == r)
    return res


entity_to_index_lookup = read_ids(entities_filename)
relation_to_index_lookup = read_ids(relations_filename)
words = read_words(words_filename)
list_of_entity_words, word_to_index_lookup = create_entity_words(entity_to_index_lookup, words)

Ne = len(entity_to_index_lookup)
Nr = len(relation_to_index_lookup)
Nw = len(words)
print("Number of entities = ", Ne)
print("Number of entities = ", Ne)
print("Number of words = ", Nw)

train_data_tuples = read_tuples(train_data_triplets_filename, entity_lookup=entity_to_index_lookup,
                                relation_lookup=relation_to_index_lookup)

# this data looks "ID-like maria_anna_of_sardinia   ID-gender  ID-female  1"
test_data_tuples = read_tuples(test_data_triplets_filename, entity_lookup=entity_to_index_lookup,
                               relation_lookup=relation_to_index_lookup, read_groud_truth=True)
dev_data_tuples = read_tuples(dev_data_triplets_filename, entity_lookup=entity_to_index_lookup,
                              relation_lookup=relation_to_index_lookup, read_groud_truth=True)

train_data = np.array(add_corrupted_exampes(train_data_tuples, C, Ne)).T
dev_data = np.array(format_dev_test_data(dev_data_tuples)).T

# print(add_corrupted_exampes(train_data, 2, len(entity_lookup)))
params = define_parameters(d, Nw, len(relation_to_index_lookup), K)
Xs, cost, optimizer, accuracy, regularization_cost = define_graph(params, Nr, K, word_to_index_lookup,
                                                                  list_of_entity_words)

print("creating data feed...")

train_data_feed = create_data_feed(Xs, train_data, Nr)
dev_data_feed = create_data_feed(Xs, dev_data, Nr)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
print("building the summary...")
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    print("writing the summaries...")
    train_writer = tf.summary.FileWriter('summary/train', sess.graph)
    test_writer = tf.summary.FileWriter('summary/test')
    print("initalizing the variables...")
    sess.run(init)
    print("starting the loop...")
    for i in range(100):

        print("   - running the train set...")
        summary, cost_value, accuracy_value, regularization_cost_value = sess.run(
            [merged, cost, accuracy, regularization_cost], feed_dict=train_data_feed)

        print("   - running the dev set...")
        dev_cost_value, dev_accuracy_value = sess.run([cost, accuracy], feed_dict=dev_data_feed)

        if i == 0:
            print("\n\n{0:10} {1:10} {2:10} {3:10} {4:10} {5:10}".format(
                "  num_iter",
                "train_cost", "train_accu",
                " dev_cost", "  dev_accu", "regulariza"))

        print("{0:10} {1:10.4} {2:10.4} {3:10.4} {4:10.4} {5:10.4}".format(
            i,
            cost_value, accuracy_value,
            dev_cost_value, dev_accuracy_value,
            regularization_cost_value))

        print("   - running optimization...")
        optimizer.minimize(sess, feed_dict=train_data_feed)

    train_writer.add_summary(summary, 1)