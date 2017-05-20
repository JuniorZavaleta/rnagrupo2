import numpy as np
import matplotlib.pyplot as plt
import csv

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

X = []
Y = []
f = open('entrenamiento.csv', 'rt')
try:
    reader = csv.reader(f)
    for row in reader:
        X.append(map(int,row[:625]))
        Y.append(map(int,row[625:]))
finally:
    f.close()


X = np.array(X)
#print X
y = np.array(Y)

#print Y
np.random.seed(1)

# randomly initialize our weights with mean 0

hidden_net = 4

syn0 = 2*np.random.random((len(X[0]),hidden_net)) - 1
syn1 = 2*np.random.random((hidden_net,len(Y[0]))) - 1

errors = []
for j in xrange(10000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    #if (j% 100) == 0:
    if np.mean(np.abs(l2_error)) < 0.01:
        errors.append(np.mean(np.abs(l2_error)))
    if errors.append(np.mean(np.abs(l2_error))) > 0.004:
        errors.append(1.0003 * np.mean(np.abs(l2_error)))
    print "Error:" + str(np.mean(np.abs(l2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

plt.plot(errors)
plt.show()
