import sys
import numpy as np
import random
import math
import subprocess

hyper_d = [100, 100] # [20, 200] # linear: doesn't matter at all
hyper_K = [4, 10] # linear
hyper_num_batches = [1, 3] # log: seems to be better on higher numbers
hyper_C = [1, 30] # linear scale
hyper_lambd = [-4, 0] #  [-3, 1]  # exponential scale, use lambd = 5*10^k



max_epochs = 4
max_runs = 200

for i in range(max_runs):
    d = int(random.uniform(*hyper_d))
    K = int(random.uniform(*hyper_K))
    num_batches = int(math.pow(10, random.uniform(*hyper_num_batches)))
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


