import numpy as np

d = 3
m = 7

np.random.seed(1)


#########################################################################################################

E1 = np.random.rand(d,m)*100
Wr = np.random.randn(d,d)
E2 = np.random.rand(d,m)*100

#########################################################################################################
expected_b = np.zeros((1,m))

for i in range(m):
	e1 = E1[:,i]
	e2 = E2[:,i]
	expected_b[0,i] = np.dot(e1.T, np.dot(Wr,e2))

print(expected_b)


#########################################################################################################

temp = np.multiply(E1, np.dot(Wr,E2))
b = np.sum(temp, axis=0, keepdims=True)

print(b)

#########################################################################################################

assert(np.squeeze(np.sum(np.square(b-expected_b))) < 1e-5 )





