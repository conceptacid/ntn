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
C = 2
threshold = 4

def read_ids(filename):
    res = {}
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            res[row[0]] = counter
            counter +=1
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
            quad = triplet + (random.randint(0, entity_size-1),)
            result.append(quad)
    return result

def format_dev_test_data(data):
    result = np.zeros((int(len(data)/2), 4))
    for i in range(0,len(data),2):
        assert(data[i][0] == data[i+1][0])
        assert(data[i][1] == data[i+1][1])
        assert(data[i][3] == 1)
        assert(data[i+1][3] == -1)
        j = (int)(i/2)
        result[j, 0] = data[i][0]   # E1
        result[j, 1] = data[i][1]   # R
        result[j, 2] = data[i][2]   # E2
        result[j, 3] = data[i+1][2] # C
    return result


def define_parameters(d=d, Ne=None, Nr=None, K=K):
    params = dict(
        E=tf.get_variable(shape=(d,Ne), name="E",  initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
    )

    for r in range(Nr):
        U=tf.get_variable(shape=(K,1), name="U_" + str(r),   initializer=tf.contrib.layers.xavier_initializer(),
                          regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
        V=tf.get_variable(shape=(K,2*d), name="V_" + str(r), initializer=tf.contrib.layers.xavier_initializer(),
                          regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
        b=tf.get_variable(shape=(K,1), name="b_" + str(r),   initializer=tf.zeros_initializer(),
                          regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
        params["U_" + str(r)] = U
        params["V_" + str(r)] = V
        params["b_" + str(r)] = b
        for k in range(K):
            name = "W_"+ str(r) + "_"+str(k)
            params[name] = tf.get_variable(shape=(d,d), name=name, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=lambd))
    return params


def build_g(params, E1, E2, r):
    SlicesArray = []
    for k in range(K):
        Wrk = params["W_" + str(r) + "_" + str(k)]

        #print("Wrk.shape=", Wrk.shape)
        #print("E2.shape=", E2.shape)

        Temp = tf.multiply(E1, tf.matmul(Wrk, E2))
        Slice = tf.reduce_sum(Temp, axis=0, keep_dims=True)

        #print("Slice.shape = ", Slice.shape)

        SlicesArray.append(Slice)
    eWe = tf.concat(SlicesArray, axis=0)

    # Vr should have shape (K,2*d)
    Vr = params["V_" + str(r)]
    assert(Vr.shape == (K,2*d))

    # br should have shape (K,1)
    br = params["b_" + str(r)]
    assert(br.shape == (K,1))

    #print("EWE.shape =", eWe.shape)

    Activation = tf.tanh(eWe + tf.matmul(Vr, tf.concat([E1, E2], axis=0)) + br) 

    g = tf.matmul(tf.transpose(params["U_" + str(r)]), Activation)
    assert(g.shape[0] == 1)
    return g


def define_graph(params, Nr, K, th):

    Xs = {}
    sums_r = []
    hits = []

    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []


    sum_gs = []
    sum_gsc = []

    for r in range(Nr):
        X = tf.placeholder(dtype=tf.int64, shape=(4,None), name="X_" + str(r))

        Xs["X_" + str(r)] = X

        E1_indices = tf.slice(X, begin=(0,0), size=(1,-1), name="E1_index_" + str(r))
        #print("E1_indices.shape=", E1_indices.shape)
        E1 = tf.gather( params["E"], E1_indices, axis=1, name="E1_" + str(r) )
        E1 = tf.reshape(E1,(d,-1))     # todo: verify
        #E1 shape should be (d,m_r)

        #print("E1.shape=", E1.shape)

        E2_indices = tf.slice(X, begin=(2,0), size=(1,-1), name="E2_index_" + str(r) )
        E2 = tf.gather( params["E"], E2_indices, axis=1, name="E2_" + str(r) )
        E2 = tf.reshape(E2, (d, -1))

        C_indices = tf.slice(X, begin=(3,0), size=(1,-1), name="C_index_" + str(r) )
        C = tf.gather( params["E"], C_indices, axis=1, name="C" + str(r)  )
        C = tf.reshape(C, (d, -1))

        g = build_g(params, E1, E2, r)
        sum_gs.append( tf.reduce_sum(g) )
        
        true_positives.append( tf.reduce_sum(tf.cast(g > th, tf.float32)) )
        false_negatives.append( tf.reduce_sum(tf.cast(g <= th, tf.float32)) )

        gc = build_g(params, E1, C, r)
        sum_gsc.append( tf.reduce_sum(gc) )

        false_positives.append( tf.reduce_sum(tf.cast(gc > th, tf.float32)) )
        true_negatives.append(  tf.reduce_sum(tf.cast(gc <= th, tf.float32)) )

        difference = tf.constant(1.) - g + gc
        max_term = tf.maximum(tf.constant(0.), difference)
        sum_r = tf.reduce_sum(max_term)
        sums_r.append(sum_r)

    
    sum_true_positives = tf.add_n(true_positives)
    sum_false_positives = tf.add_n(false_positives)
    sum_true_negatives = tf.add_n(true_negatives)
    sum_false_negatives = tf.add_n(false_negatives)

    avg_g = tf.add_n(sum_gs)/(sum_true_positives + sum_false_negatives)
    avg_gc = tf.add_n(sum_gsc)/(sum_false_positives + sum_true_negatives)

    accuracy = (sum_true_positives + sum_true_negatives)/(sum_true_positives + sum_true_negatives + sum_false_positives + sum_false_negatives)


    cost = tf.add_n(sums_r) #+ tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    tf.summary.scalar('cost', cost)

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(cost, options={'maxiter' : 1})  # if 'method' arg is undefined, the default method is L-BFGS-B

    return Xs, cost, optimizer, accuracy, avg_g, avg_gc


def create_data_feed(Xs, quadruplets, Nr):
    res = {}
    for r in range(Nr):
        name = "X_" + str(r)
        res[Xs[name]] = quadruplets[:, quadruplets[1, :] == r]
        print(" data feed X_",r, " is ", res[Xs[name]].shape, res[Xs[name]].dtype )
        #print("r=",r, "\n", quadruplets[1, :] == r)
    return res


entity_lookup = read_ids(entities_filename)
relation_lookup = read_ids(relations_filename)

Ne = len(entity_lookup)
Nr = len(relation_lookup)
print("Number of entities = ", Ne)
print("Number of relations = ", Nr)

train_data_tuples = read_tuples(train_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup)


# this data looks "ID-like maria_anna_of_sardinia   ID-gender  ID-female  1"
test_data_tuples = read_tuples(test_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup, read_groud_truth=True)
dev_data_tuples = read_tuples(dev_data_triplets_filename, entity_lookup=entity_lookup, relation_lookup=relation_lookup, read_groud_truth=True)

train_data = np.array(add_corrupted_exampes(train_data_tuples, C, Ne)).T
dev_data = np.array(format_dev_test_data(dev_data_tuples)).T

#print(add_corrupted_exampes(train_data, 2, len(entity_lookup)))
params = define_parameters(Ne=Ne, Nr=len(relation_lookup))
Xs, cost, optimizer, accuracy, avg_g, avg_gc = define_graph(params, Nr, K, threshold)

train_data_feed = create_data_feed(Xs, train_data, Nr)
dev_data_feed = create_data_feed(Xs, dev_data, Nr)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()



with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('summary/train', sess.graph)
    test_writer = tf.summary.FileWriter('summary/test')
    sess.run(init)
    
    for i in range(30):
        summary, cost_value, accuracy_value, avg_g_value, avg_gc_value = sess.run([merged, cost, accuracy, avg_g, avg_gc], feed_dict=train_data_feed)

        dev_cost_value, dev_accuracy_value, dev_avg_g_value, dev_avg_gc_value = sess.run([cost, accuracy, avg_g, avg_gc], feed_dict=dev_data_feed)

        if i == 0:
            print("\n\n{0:10} {1:10} {2:10} {3:10} {4:10} {5:10} {6:10} {7:10} {8:10}".format(
                 "num_iter",
                 "train_cost", "train_accu", "train_g", "train_gc",    
                  "dev_cost", "dev_accu", "dev_g", "dev_gc"))


        print("{0:10} {1:10.4} {2:10.4} {3:10.4} {4:10.4} {5:10.4} {6:10.4} {7:10.4} {8:10.4}".format(
            i, 
            cost_value, accuracy_value, avg_g_value, avg_gc_value,
            dev_cost_value, dev_accuracy_value, dev_avg_g_value, dev_avg_gc_value))

        optimizer.minimize(sess, feed_dict=train_data_feed)
    
    train_writer.add_summary(summary, 1)