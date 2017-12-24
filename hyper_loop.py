import sys
import numpy as np
import random
import math
import subprocess

hyper_d = [20, 200] # linear
hyper_K = [2, 8] # linear
hyper_num_batches = [0, 2]
hyper_C = [1, 20] # linear scale
hyper_lambd = [-3, 1]  # exponential scale, use lambd = 5*10^k

max_epochs = 1
max_runs = 100

for i in range(max_runs):
    d = int(random.uniform(*hyper_d))
    K = int(random.uniform(*hyper_K))
    num_batches = int(5 + math.pow(10, random.uniform(*hyper_num_batches)))
    C = int(random.uniform(*hyper_C))
    lambd = float(5*math.pow(10, random.uniform(*hyper_lambd)))

    args = [str(d), str(K), str(num_batches), str(C), str(lambd), str(max_epochs)]

    cmd =["python3", "train_word_vectors_with_minibatches.py"] + args

    print(' starting', i+1, 'of', max_runs, 'args =', ' '.join(args))

    suffix = "_".join(args) + ".txt"
    with open("stdout_"+suffix, "w") as out, open("stderr_"+suffix, "w") as err:
        p = subprocess.Popen(cmd, stdout=out, stderr=err)
        res = p.wait()
        print('   result code', res)


