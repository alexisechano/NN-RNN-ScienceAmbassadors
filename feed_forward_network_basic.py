#   Feed Forward Lab for Artificial Intelligence, Language: Python
#   Created and Programmed by Alexis Echano 
#    
#   PURPOSE: demonstrates how a basic neural network can take in inputs along with pre-set weights to calculate a basic math equation

#   Importing all relevant packages to run the program
import os, sys, math

#   A map of the layer number and associated weights in an attached list
WEIGHTS = {0: [[0.2770149726270103, 0.08599525311594647, 0.1923150548201215],[0.15822995838275988, 0.3125973713795491, 0.7585852143863143]],
          1: [[0.09050538657268319, 0.9295545446844707]], 2: [[0.8159477905497953]]}

#   This is a sort of directory for the different transfer functions which are the neuron's task to carry out
def transfer_function(num, sum_val):   #num is the T#
    if num == 1: return transfunc_one(sum_val)
    elif num == 2: return transfunc_two(sum_val)
    elif num == 3: return transfunc_three(sum_val)
    else: return transfunc_four(sum_val)

#   A very basic transfer function that return the value given to the particular neuron
def transfunc_one(val): #basic
    return val

#   This outlines a basic ramp function so if the value given to the neuron is negative, it will return 0. If not, it will return the value, like above
def transfunc_two(val): #ramp
    if val >= 0:
        return val
    return 0.0

#   This is a logistic calcluation, most common in neural networks to use this
def transfunc_three(val):   
    return 1.0/(1.0 + (math.e**(-1.0 * val)))

#   The last transfer function is just doubling the logistic one, almost deriving it
def transfunc_four(val):    
    return (2.0 * transfunc_three(val)) - 1.0

#   A lot of complicated code here, but it takes in a text file and constructs Python-friendly data structures that we can work with
def read_file(file_list, leng):   
    global WEIGHTS
    layer_num = 0   #keep track of which layer

    #leng var is the length of inputs so we can determine values
    weights_per_layer = []
    weights_a_node = []

    #the next value is the len of the previous dict entry!!
    for line in file_list:  #reads line by line, index in value is the transfer node's assignment
        line = line.split()
        if layer_num == 0:
            i = 1
            for num in line:
                num = float(num)
                weights_a_node.append(num)
                if i < leng:
                    i += 1
                else:
                    weights_per_layer.append(weights_a_node)
                    weights_a_node = []
                    i = 1
        else:
            length_new = len(WEIGHTS[layer_num-1])
            x = 1
            for val in line:
                val = float(val)
                weights_a_node.append(val)
                if x < length_new:
                    x += 1
                else:
                    weights_per_layer.append(weights_a_node)
                    weights_a_node = []
                    x = 1

        WEIGHTS[layer_num] = weights_per_layer
        weights_per_layer = []
        layer_num += 1

#   The dot product is the combination of weights, input values, and transfer function outputs that will be sent to the next layer
def dot_product_sum(inputs, weight_list):  
    return_lists = []   #sums, index is the weights
    weighted = []

    for node in weight_list:    # node is a list of weights for the next layer
        for x in range(len(node)):
            temp_val = inputs[x] * node[x]
            weighted.append(temp_val)
        return_lists.append(sum(i for i in weighted))
        weighted = []

    return return_lists

#   The main method that runs the entire program
def main():
  
    input_vals = [0.0, 0.0, 1.0] #   no need for the read file as I have hardcoded some input values
    transNum = -1

    transNum = int(3)  #chooses the transfer function, in this case, the logistic one

    layer = 0

    new_inputs = input_vals #put transfer functioned, and added values here for next layer

    while layer < len(WEIGHTS) - 1:
        to_be_trans = dot_product_sum(new_inputs, WEIGHTS[layer])
        new_inputs = []

        for new_node in to_be_trans:
            new_inputs.append(transfer_function(transNum, new_node))

        layer += 1

    #   final layer - I just programmed this to calculate by "hand"
    final_weights = WEIGHTS[layer][0]
    result = []
    for i in range(len(new_inputs)):
        value = new_inputs[i] * final_weights[i]
        result.append(value)

    #   Prints the output result
    for thing in result:
        print(thing, end=" ")
    print()

    print(.5*(0.0-thing)**2)  #error to see how close the expected value (0.0) is to the NN's value

#   run main file
main()
