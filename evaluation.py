from nimblenet.activation_functions import sigmoid_function
from nimblenet.activation_functions import softmax_function
from nimblenet.cost_functions import cross_entropy_cost
from nimblenet.cost_functions import softmax_neg_loss

from nimblenet.data_structures import Instance
from nimblenet.neuralnet import NeuralNet
import csv

import numpy as np

import matplotlib.pyplot as plt

from nimblenet.tools import print_test

import collections
import random
import math

errors = []

default_configuration = {
    'ERROR_LIMIT'           : 0.001, 
    'learning_rate'         : 0.03, 
    'batch_size'            : 1, 
    'print_rate'            : 1000, 
    'save_trained_network'  : False,
    'input_layer_dropout'   : 0.0,
    'hidden_layer_dropout'  : 0.0, 
    'evaluation_function'   : None,
    'max_iterations'        : ()
}

def dropout( X, p = 0. ):
    if p != 0:
        retain_p = 1 - p
        X = X * np.random.binomial(1,retain_p,size = X.shape)
        X /= retain_p
    return X
#end  

def add_bias(A):
    # Add a bias value of 1. The value of the bias is adjusted through
    # weights rather than modifying the input signal.
    return np.hstack(( np.ones((A.shape[0],1)), A ))
#end addBias


def confirm( promt='Do you want to continue?' ):
    prompt = '%s [%s|%s]: ' % (promt,'y','n')
    while True:
        ans = raw_input(prompt).lower()
        if ans in ['y','yes']:
            return True
        if ans in ['n','no']:
            return False
        print "Please enter y or n."
#end

def verify_dataset_shape_and_modify( network, dataset ):   
    assert dataset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert dataset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    data              = np.array( [instance.features for instance in dataset ] )
    targets           = np.array( [instance.targets  for instance in dataset ] )
    
    return data, targets 
#end


def apply_regularizers( dataset, cost_function, regularizers, network ):
    dW_regularizer = lambda x: np.zeros( shape = x.shape )
    
    if regularizers != None:
        # Modify the cost function to add the regularizer
        for entry in regularizers:
            if type(entry) == tuple:
                regularizer, regularizer_settings = entry
                cost_function, dW_regularizer  = regularizer( dataset, cost_function, dW_regularizer, network, **regularizer_settings )
            else:
                regularizer    = entry
                cost_function, dW_regularizer  = regularizer( dataset, cost_function, dW_regularizer, network )
    
    return cost_function, dW_regularizer
#end


def check_network_structure( network, cost_function ):
    assert softmax_function != network.layers[-1][1] or cost_function == softmax_neg_loss,\
        "When using the `softmax` activation function, the cost function MUST be `softmax_neg_loss`."
    assert cost_function != softmax_neg_loss or softmax_function == network.layers[-1][1],\
        "When using the `softmax_neg_loss` cost function, the activation function in the final layer MUST be `softmax`."
#end

def backpropagation_foundation(network, trainingset, testset, cost_function, calculate_dW, evaluation_function = None, ERROR_LIMIT = 1e-3, max_iterations = (), batch_size = 0, input_layer_dropout = 0.0, hidden_layer_dropout = 0.0, print_rate = 1000, save_trained_network = False, **kwargs):
    check_network_structure( network, cost_function ) # check for special case topology requirements, such as softmax
    
    training_data, training_targets = verify_dataset_shape_and_modify( network, trainingset )
    test_data, test_targets    = verify_dataset_shape_and_modify( network, testset )
    
    
    # Whether to use another function for printing the dataset error than the cost function. 
    # This is useful if you train the network with the MSE cost function, but are going to 
    # classify rather than regress on your data.
    if evaluation_function != None:
        calculate_print_error = evaluation_function
    else:
        calculate_print_error = cost_function
    
    batch_size                 = batch_size if batch_size != 0 else training_data.shape[0] 
    batch_training_data        = np.array_split(training_data, math.ceil(1.0 * training_data.shape[0] / batch_size))
    batch_training_targets     = np.array_split(training_targets, math.ceil(1.0 * training_targets.shape[0] / batch_size))
    batch_indices              = range(len(batch_training_data))       # fast reference to batches
    
    error                      = calculate_print_error(network.update( test_data ), test_targets )
    reversed_layer_indexes     = range( len(network.layers) )[::-1]
    
    epoch                      = 0
    while error > ERROR_LIMIT and epoch < max_iterations:
        epoch += 1
        
        random.shuffle(batch_indices) # Shuffle the order in which the batches are processed between the iterations
        
        for batch_index in batch_indices:
            batch_data                 = batch_training_data[    batch_index ]
            batch_targets              = batch_training_targets[ batch_index ]
            batch_size                 = float( batch_data.shape[0] )
            
            input_signals, derivatives = network.update( batch_data, trace=True )
            out                        = input_signals[-1]
            cost_derivative            = cost_function( out, batch_targets, derivative=True ).T
            delta                      = cost_derivative * derivatives[-1]
            
            for i in reversed_layer_indexes:
                # Loop over the weight layers in reversed order to calculate the deltas
            
                # perform dropout
                dropped = dropout( 
                            input_signals[i], 
                            # dropout probability
                            hidden_layer_dropout if i > 0 else input_layer_dropout
                        )
            
                # calculate the weight change
                dX = (np.dot( delta, add_bias(dropped) )/batch_size).T
                dW = calculate_dW( i, dX )
                
                if i != 0:
                    """Do not calculate the delta unnecessarily."""
                    # Skip the bias weight
                    weight_delta = np.dot( network.weights[ i ][1:,:], delta )
    
                    # Calculate the delta for the subsequent layer
                    delta = weight_delta * derivatives[i-1]
                
                # Update the weights with Nestrov Momentum
                network.weights[ i ] += dW
            #end weight adjustment loop
        
        error = calculate_print_error(network.update( test_data ), test_targets )
        errors.append(np.mean(np.abs(error)))
        if epoch%print_rate==0:
            # Show the current training status
            print "[training] Current error:", error, "\tEpoch:", epoch
    
    print "[training] Finished:"
    print "[training]   Converged to error bound (%.4g) with error %.4g." % ( ERROR_LIMIT, error )
    print "[training]   Measured quality: %.4g" % network.measure_quality( training_data, training_targets, cost_function )
    print "[training]   Trained for %d epochs." % epoch
    
    if save_trained_network and confirm( promt = "Do you wish to store the trained network?" ):
        network.save_network_to_file()
# end backprop

def RMSprop2(network, trainingset, testset, cost_function, decay_rate = 0.99, epsilon = 1e-8, **kwargs  ):
    configuration = dict(default_configuration)
    configuration.update( kwargs )
    
    learning_rate = configuration["learning_rate"]
    cache         = [ np.zeros( shape = weight_layer.shape ) for weight_layer in network.weights ]
    
    def calculate_dW( layer_index, dX ):
        cache[ layer_index ] = decay_rate * cache[ layer_index ] + (1 - decay_rate) * dX**2
        return -learning_rate * dX / (np.sqrt(cache[ layer_index ]) + epsilon)
    #end
    
    return backpropagation_foundation( network, trainingset, testset, cost_function, calculate_dW, **configuration )
#end

#dataset = [
#    Instance( [0,0,1], [0] ),
#    Instance( [0,1,1], [1] ),
#    Instance( [1,0,1], [1] ),
#    Instance( [1,1,1], [1] )
#]



def print_test2( network, testset, cost_function ):
    assert testset[0].features.shape[0] == network.n_inputs, \
        "ERROR: input size varies from the defined input setting"
    assert testset[0].targets.shape[0]  == network.layers[-1][0], \
        "ERROR: output size varies from the defined output setting"
    
    test_data              = np.array( [instance.features for instance in testset ] )
    test_targets           = np.array( [instance.targets  for instance in testset ] )
    
    input_signals, derivatives = network.update( test_data, trace=True )
    out                        = input_signals[-1]
    error                      = cost_function(out, test_targets )
    
    print "[testing] Network error: %.4g" % error
    print "[testing] Network results:"
    print "[testing]   input\tresult\ttarget"
    for entry, result, target in zip(test_data, out, test_targets):
        print "[testing]   %s\t%s\t%s" % tuple(map(str, [entry, result, target]))
#end

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
    "layers"   : [  (666, softmax_function) , (row_len, softmax_function) ],

}

network        = NeuralNet( settings )
cost_function  = softmax_neg_loss


RMSprop2(network, training_set, test_set,cost_function, learning_rate = 0.01, print_rate = 10, ERROR_LIMIT = 0.2)

#print_test( network, training_set, cost_function )


"""
Prediction Example
"""
prediction_set = test_set
prediction_set = prediction_set
print network.predict( prediction_set ) # produce the output signal
print len(errors)
plt.plot(errors)
plt.show()