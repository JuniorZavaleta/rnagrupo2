from nimblenet.activation_functions import sigmoid_function
from nimblenet.activation_functions import softmax_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.cost_functions import softmax_neg_loss
from nimblenet.learning_algorithms import RMSprop
from nimblenet.data_structures import Instance
from nimblenet.neuralnet import NeuralNet
import csv

#dataset = [
#    Instance( [0,0,1], [0] ),
#    Instance( [0,1,1], [1] ),
#    Instance( [1,0,1], [1] ),
#    Instance( [1,1,1], [1] )
#]

training_set = []
test_set = []

row_len = 0
col_len = 0
iteration = 0

f = open('entrenamiento.csv', 'rt')
try:
    reader = csv.reader(f)
    for row in reader:
        X = (map(int,row[:625]))
        Y = map(int,row[625:])
        col_len = len(X)
        row_len = len(Y)
        training_set.append(Instance(X, Y))
        iteration = iteration + 1
        #if iteration == 12:
            #break
finally:
    f.close()

g = open('test.csv', 'rt')
try:
    reader = csv.reader(g)
    for row in reader:
        X = (map(int,row[:625]))
        Y = map(int,row[625:])
        test_set.append(Instance(X, Y))
finally:
    g.close()

settings       = {
    "n_inputs" : col_len,
    "layers"   : [  (1000, softmax_function) , (row_len, softmax_function) ],

}

network        = NeuralNet( settings )
cost_function  = softmax_neg_loss


RMSprop(network, training_set, test_set,cost_function, print_rate = 10)
