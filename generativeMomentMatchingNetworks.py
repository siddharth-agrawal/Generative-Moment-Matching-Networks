import argparse
import cPickle as pickle
import math
import numpy as np
import random
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

"""
Give the training images from the MNIST dataset
"""
def loadMNIST():

    # Downloaded from http://deeplearning.net/data/mnist/mnist.pkl.gz
    train_data, val_data, test_data = pickle.load(open('mnist.pkl', 'r'))
    train_x, train_y = train_data

    return train_x

"""
Give the training images from the cropped LFW dataset
"""
def loadLFW():

    # 32x32 version of grayscale cropped LFW
    # Original dataset here: http://conradsanderson.id.au/lfwcrop/
    return np.load('lfw.npy')

"""
Return a TF variable with zeros of provided shape
"""
def zeros(shape):

    return tf.Variable(tf.zeros(shape))

"""
Return a TF variable with numbers drawn from a normal distribution of zero mean
and given standard deviation
"""
def normal(shape, std_dev):

    return tf.Variable(tf.random_normal(shape, stddev = std_dev))

class ReLULayer():

    """
    Initialize layer object with the given input and output dimensions

    input_dim:  Dimension of inputs to the layer
    output_dim: Dimension of outputs of the layer
    """
    def __init__(self, input_dim, output_dim):

        # initialize weights and biases for the layer
        self.W = normal([input_dim, output_dim], 1.0 / math.sqrt(input_dim))
        self.b = zeros([output_dim])

    """
    Forward propagation in the layer

    x: Input to the layer
    """
    def forward(self, x):

        return tf.nn.relu(tf.matmul(x, self.W) + self.b)

class SigmoidLayer():

    """
    Initialize layer object with the given input, output dimensions and dropout
    retention probabilities

    input_dim:    Dimension of inputs to the layer
    output_dim:   Dimension of outputs of the layer
    dropout_prob: Fraction of dropout retention in the layer
    """
    def __init__(self, input_dim, output_dim, dropout_prob = 1.0):

        # initialize weights and biases for the layer
        self.W = normal([input_dim, output_dim], 1.0 / math.sqrt(input_dim))
        self.b = zeros([output_dim])

        # store the dropout retention probability for later use
        self.dropout_prob = dropout_prob

    """
    Forward propagation in the layer

    x: Input to the layer
    """
    def forward(self, x):

        return tf.sigmoid(tf.matmul(tf.nn.dropout(x, self.dropout_prob),
                          self.W) + self.b)

class DataSpaceNetwork():

    """
    Initialize network object with the given dimensions and batch size

    dimensions: Dimensions of the all the layers of the network, including
                input and output
    batch_size: Number of training examples taken in the batch
    """
    def __init__(self, dimensions, batch_size):

        # store 'dimensions' and 'batch_size' for later use
        self.dimensions = dimensions
        self.batch_size = batch_size

        # store the layers as a list
        self.layers = []

        # all the layers except the last one is 'ReLU'
        for dim_index in range(len(dimensions)-2):
            self.layers.append(ReLULayer(dimensions[dim_index],
                                         dimensions[dim_index+1]))

        # last layer is 'Sigmoid' as we need the outputs to be in [0, 1]
        self.layers.append(SigmoidLayer(dimensions[dim_index+1],
                                        dimensions[dim_index+2]))

    """
    Forward propagation of the network

    x: Input batch of samples from the uniform
    """
    def forward(self, x):

        # initialize the first 'hidden' layer to the input
        h = x

        # for all the layers propagate the activation forward
        # all layers have the 'forward()' method
        for dim_index in range(len(self.dimensions)-1):
            h = self.layers[dim_index].forward(h)

        return h

    """
    Scale column for the MMD measure

    num_gen:  Number of samples to be generated in one pass, 'N' in the paper
    num_orig: Number of samples taken from dataset in one pass, 'M' in the paper
    """
    def makeScaleMatrix(self, num_gen, num_orig):

        # first 'N' entries have '1/N', next 'M' entries have '-1/M'
        s1 =  tf.constant(1.0 / num_gen, shape = [num_gen, 1])
        s2 = -tf.constant(1.0 / num_orig, shape = [num_orig, 1])

        return tf.concat(0, [s1, s2])

    """
    Calculates cost of the network, which is square root of the mixture of 'K'
    RBF kernels

    x:       Batch from the dataset
    samples: Samples from the uniform distribution
    sigma:   Bandwidth parameters for the 'K' kernels
    """
    def computeLoss(self, x, samples, sigma = [2, 5, 10, 20, 40, 80]):

        # generate images from the provided uniform samples
        gen_x = self.forward(samples)

        # concatenation of the generated images and images from the dataset
        # first 'N' rows are the generated ones, next 'M' are from the data
        X = tf.concat(0, [gen_x, x])

        # dot product between all combinations of rows in 'X'
        XX = tf.matmul(X, tf.transpose(X))

        # dot product of rows with themselves
        X2 = tf.reduce_sum(X * X, 1, keep_dims = True)

        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
        exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

        # scaling constants for each of the rows in 'X'
        s = self.makeScaleMatrix(self.batch_size, self.batch_size)

        # scaling factors of each of the kernel values, corresponding to the
        # exponent values
        S = tf.matmul(s, tf.transpose(s))

        loss = 0

        # for each bandwidth parameter, compute the MMD value and add them all
        for i in range(len(sigma)):

            # kernel values for each combination of the rows in 'X' 
            kernel_val = tf.exp(1.0 / sigma[i] * exponent)
            loss += tf.reduce_sum(S * kernel_val)

        return tf.sqrt(loss)

class Autoencoder():

    """
    Initialize autoencoder with the given dimensions and dropout fractions for
    each of the layers

    dimensions: Dimensions of the autoencoder from the input till the innermost
                hidden layer
    dropout:    Retention fractions for dropout in the hidden layers
    """
    def __init__(self, dimensions, dropout):

        # store 'dimensions' for later use
        self.dimensions = dimensions

        # store the layers as a list
        self.layers = []

        # add the encoder layers
        for dim_index in range(len(dimensions)-1):
            self.layers.append(SigmoidLayer(dimensions[dim_index],
                                            dimensions[dim_index+1],
                                            dropout[dim_index]))

        # add the decoder layers
        for dim_index in range(len(dimensions)-1)[::-1]:
            self.layers.append(SigmoidLayer(dimensions[dim_index+1],
                                            dimensions[dim_index]))

    """
    Reconstruction cost for one layer

    x:           Input batch of images
    layer_index: Index of the layer to train
    """
    def layerCost(self, x, layer_index):

        # initialize the input representation to the passed images
        input_rep = x

        # get the input representation to this layer by forward propagating on
        # the previously trained layers
        for layer in range(layer_index):
            input_rep = self.layers[layer].forward(input_rep)

        # get the hidden representation for the layer
        h = self.layers[layer_index].forward(input_rep)

        # reconstruct using the hidden representation
        rec = self.layers[len(self.layers) - 1 - layer_index].forward(h)

        # return the cross entropy loss between the input representation and the
        # reconstruction
        return -tf.reduce_sum(input_rep * tf.log(rec) + (1 - input_rep) *
                              tf.log(1 - rec))

    """
    Reconstruction cost using the network of stacked autoencoders

    x: Input batch of images
    """
    def finetuneCost(self, x):

        # initialize hidden representation to the input
        h = x

        # forward propagation over all the layers
        for layer in range(len(self.layers)):
            h = self.layers[layer].forward(h)

        # return the cross entropy between the input images and the
        # reconstruction
        return -tf.reduce_sum(x * tf.log(h) + (1 - x) * tf.log(1 - h))

class CodeSpaceNetwork():

    """
    Initialize the network with the given dimensions, autoencoder and the
    batch size

    dimensions:   Dimensions of the all the layers of the network, including
                  input and output
    auto_encoder: The autoencoder object to be used in the code space network
    batch_size:   Number of training examples taken in the batch
    """
    def __init__(self, dimensions, auto_encoder, batch_size):

        # store 'dimensions', 'auto_encoder' and 'batch_size' for later use
        self.dimensions   = dimensions
        self.auto_encoder = auto_encoder
        self.batch_size   = batch_size

        # store the network layers as a list
        self.layers = []

        # all the layers except the last one is 'ReLU'
        for dim_index in range(len(dimensions)-1):
            self.layers.append(ReLULayer(dimensions[dim_index],
                                         dimensions[dim_index+1]))

        # dimenison of the codes to be generated
        decoder_input_size = auto_encoder.dimensions[-1]

        # the last layer is 'Sigmoid' as all the layers of the autoencoder are
        # 'Sigmoid'
        self.layers.append(SigmoidLayer(dimensions[dim_index+1],
                                        decoder_input_size))

    """
    Forward propagation of the network

    x: Input batch of samples from the uniform
    """
    def forward(self, x):

        # initialize the first 'hidden' layer to the input
        h = x

        # for all the layers propagate the activation forward
        # all layers have the 'forward()' method
        for dim_index in range(len(self.dimensions)):
            h = self.layers[dim_index].forward(h)

        return h

    """
    Generation of image samples from the network

    x: Input batch of samples from the uniform
    """
    def generate(self, x):

        # generate codes from the uniform samples
        h = x
        for dim_index in range(len(self.dimensions)):
            h = self.layers[dim_index].forward(h)

        # start layer of the decoder of the autoencoder
        layer_index = len(self.auto_encoder.dimensions) - 1

        # generate images using the above generated codes
        while layer_index < len(self.auto_encoder.layers):
            h = self.auto_encoder.layers[layer_index].forward(h)
            layer_index += 1

        return h

    """
    Encode the input images

    x: Input batch of images from the dataset
    """
    def encode(self, x):

        # initialize the 'hidden' layer to the input
        h = x

        # start layer of the encoder
        layer_index = 0

        # propagate forward till the innermost layer
        while layer_index < len(self.auto_encoder.layers)/2:
            h = self.auto_encoder.layers[layer_index].forward(h)
            layer_index += 1

        # stop the gradient as we don't want to train the autoencoder while
        # training the network
        return tf.stop_gradient(h)

    """
    Scale column for the MMD measure

    num_gen:  Number of samples to be generated in one pass, 'N' in the paper
    num_orig: Number of samples taken from dataset in one pass, 'M' in the paper
    """
    def makeScaleMatrix(self, num_gen, num_orig):

        # first 'N' entries have '1/N', next 'M' entries have '-1/M'
        s1 =  tf.constant(1.0 / num_gen, shape = [num_gen, 1])
        s2 = -tf.constant(1.0 / num_orig, shape = [num_orig, 1])

        return tf.concat(0, [s1, s2])

    """
    Calculates cost of the network, which is square root of the mixture of 'K'
    RBF kernels

    x:       Batch from the dataset
    samples: Samples from the uniform distribution
    sigma:   Bandwidth parameters for the 'K' kernels
    """
    def computeLoss(self, x, samples, sigma = [1]):

        # generate codes from the uniform samples
        gen_x = self.forward(samples)

        # generate autoencoder codes from the dataset batch
        encode_x = self.encode(x)

        # concatenation of the generated codes and the autoencoder codes for
        # batch of images from the dataset
        X = tf.concat(0, [gen_x, encode_x])

        # dot product between all combinations of rows in 'X'
        XX = tf.matmul(X, tf.transpose(X))

        # dot product of rows with themselves
        X2 = tf.reduce_sum(X * X, 1, keep_dims = True)

        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
        exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

        # scaling constants for each of the rows in 'X'
        s = self.makeScaleMatrix(self.batch_size, self.batch_size)

        # scaling factors of each of the kernel values, corresponding to the
        # exponent values
        S = tf.matmul(s, tf.transpose(s))

        loss = 0

        # for each bandwidth parameter, compute the MMD value and add them all
        for i in range(len(sigma)):

            # kernel values for each combination of the rows in 'X' 
            kernel_val = tf.exp(1.0 / sigma[i] * exponent)
            loss += tf.reduce_sum(S * kernel_val)

        return tf.sqrt(loss)

"""
Generate figure of the given generated samples

samples:    Samples generated by the network
num_rows:   Number of rows in the generated figure
num_cols:   Number of columns in the generated figure
image_side: Width and height of a single image in the figure
file_name:  File name for the generated figure to be saved
"""
def generateFigure(samples, num_rows, num_cols, image_side, file_name):

    # initialize the figure object
    figure, axes = plt.subplots(nrows = num_rows, ncols = num_cols)

    index = 0
    # take the first 'num_rows * num_cols' samples from the provided batch
    for axis in axes.flat:
        image = axis.imshow(samples[index, :].reshape(image_side, image_side),
                            cmap = plt.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1 

    # save the figure
    figure.savefig(file_name)

"""
Train data space network on the given dataset

dataset: Either 'mnist' or 'lfw', indicating the dataset
"""
def trainDataSpaceNetwork(dataset):

    # batch size for the training
    batch_size = 1000

    # parameters and training set for MNIST
    if dataset == 'mnist':
        input_dim    = 784
        image_side   = 28
        num_examples = 50000
        train_x      = loadMNIST()

    # parameters and training set for LFW
    elif dataset == 'lfw':
        input_dim    = 1024
        image_side   = 32
        num_examples = 13000
        train_x      = loadLFW()

    # dimensions of the moment matching network
    data_space_dims = [10, 64, 256, 256, input_dim]

    # get a DataSpaceNetwork object
    data_space_network = DataSpaceNetwork(data_space_dims, batch_size)

    # placeholders for the data batch and the uniform samples respectively
    x       = tf.placeholder("float", [batch_size, input_dim])
    samples = tf.placeholder("float", [batch_size, data_space_dims[0]])

    # cost of the network, and optimizer for the cost
    cost      = data_space_network.computeLoss(x, samples)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # generator for the network
    generate = data_space_network.forward(samples)

    # initalize all the variables in the model
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # number of batches to train the model on, and frequency of printing out the
    # cost
    num_iterations  = 40001
    iteration_break = 1000

    for i in range(num_iterations):

        # sample a random batch from the training set
        batch_indices = np.random.randint(num_examples, size = batch_size)
        batch_x       = train_x[batch_indices]
        batch_uniform = np.random.uniform(low = -1.0, high = 1.0,
            size = (batch_size, data_space_dims[0]))

        # print out the cost after every 'iteration_break' iterations
        if i % iteration_break == 0:
            curr_cost = sess.run(cost, feed_dict = {samples: batch_uniform,
                                                    x: batch_x})
            print 'Cost at iteration ' + str(i+1) + ': ' + str(curr_cost)

        # optimize the network
        sess.run(optimizer, feed_dict = {samples: batch_uniform, x: batch_x})

    # parameters for figure generation
    num_rows = 10; num_cols = 10

    # generate samples from the trained network
    batch_uniform = np.random.uniform(low = -1.0, high = 1.0,
        size = (batch_size, data_space_dims[0]))
    gen_samples   = sess.run(generate, feed_dict = {samples: batch_uniform})

    # generate figure of generated samples
    file_name = dataset + '_data_space.png'
    generateFigure(gen_samples, num_rows, num_cols, image_side, file_name)

"""
Train code space network on the given dataset

dataset: Either 'mnist' or 'lfw', indicating the dataset
"""
def trainCodeSpaceNetwork(dataset):

    # batch size for training autoencoder
    enc_batch_size = 100
    # batch size for training moment matching network
    batch_size     = 1000

    # parameters and training set for MNIST
    if dataset == 'mnist':
        input_dim    = 784
        image_side   = 28
        num_examples = 50000
        train_x      = loadMNIST()

    # parameters and training set for LFW
    elif dataset == 'lfw':
        input_dim    = 1024
        image_side   = 32
        num_examples = 13000
        train_x      = loadLFW()

    # dimensions for the encoder; decoder dimensions are implicit
    auto_encoder_dims = [input_dim, 1024, 32]
    # dimensions of the moment matching network
    code_space_dims   = [10, 64, 256, 256, input_dim]

    # get Autoencoder and CodeSpaceNetwork objects
    auto_encoder       = Autoencoder(auto_encoder_dims, [0.8, 0.5])
    code_space_network = CodeSpaceNetwork(code_space_dims, auto_encoder,
                                          batch_size)

    # placeholders for data batches (autoencoder and moment matching network)
    # and the uniform samples
    x_enc   = tf.placeholder("float", [enc_batch_size, input_dim])
    x_code  = tf.placeholder("float", [batch_size, input_dim])
    samples = tf.placeholder("float", [batch_size, code_space_dims[0]])

    # lists for the layer costs and their optimizers (for the autoencoder)
    layer_costs      = []
    layer_costs_opts = []

    # for every layer append the layer cost and its optimizer
    for layer_index in range(len(auto_encoder_dims)-1):

        layer_cost = auto_encoder.layerCost(x_enc, layer_index)
        layer_costs.append(layer_cost)
        layer_costs_opts.append(tf.train.AdamOptimizer().minimize(layer_cost))

    # finetuning cost for the autoencoder and its optimizer
    finetune_cost = auto_encoder.finetuneCost(x_enc)
    finetune_opt  = tf.train.AdamOptimizer().minimize(finetune_cost)

    # cost for the moment matching network and its optimizer
    code_space_cost = code_space_network.computeLoss(x_code, samples)
    code_optimizer  = tf.train.AdamOptimizer().minimize(code_space_cost)

    # generator for the network
    generate = code_space_network.generate(samples)

    # initialize all the variables in the model
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # number of batches to train the each layer on, and frequency of printing
    # out the cost
    num_iterations  = 3001
    iteration_break = 100

    # greedily optimize each layer
    for layer_index in range(len(auto_encoder_dims)-1):

        # cost of the current layer and its optimizer
        curr_layer_cost = layer_costs[layer_index]
        optimizer       = layer_costs_opts[layer_index]

        for i in range(num_iterations):

            # sample a random batch from the training set
            batch_indices = np.random.randint(num_examples,
                                              size = enc_batch_size)
            batch_x = train_x[batch_indices, :]

            # print out the cost after every 'iteration_break' iterations
            if i % iteration_break == 0:
                curr_cost = sess.run(curr_layer_cost,
                                     feed_dict = {x_enc: batch_x})
                print 'Autoencoder' + str(layer_index+1) + \
                    ' cost at iteration ' + str(i+1) + ': ' + str(curr_cost)

            # optimize the layer
            sess.run(optimizer, feed_dict = {x_enc: batch_x})

    # number of batches to finetune the autoencoder on
    num_iterations  = 4001

    # finetune the autoencoder
    for i in range(num_iterations):

        # sample a random batch from the training set and finetune the
        # autoencoder
        batch_indices = np.random.randint(num_examples, size = enc_batch_size)
        batch_x       = train_x[batch_indices, :]
        sess.run(finetune_opt, feed_dict = {x_enc: batch_x})

        # print out the cost after every 'iteration_break' iterations
        if i % iteration_break == 0:
            curr_cost = sess.run(finetune_cost, feed_dict = {x_enc: batch_x})
            print 'Stacked autoencoder cost at iteration ' + str(i+1) + ': ' + \
                str(curr_cost)

    # number of batches to train the moment matching network on, and frequency
    # of printing out the cost
    num_iterations  = 40001
    iteration_break = 1000

    for i in range(num_iterations):

        # sample a random batch from the training set, batch of uniform samples
        batch_indices = np.random.randint(num_examples, size = batch_size)
        batch_x       = train_x[batch_indices, :]
        batch_uniform = np.random.uniform(low = -1.0, high = 1.0,
            size = (batch_size, code_space_dims[0]))

        # print out the cost after every 'iteration_break' iterations
        if i % iteration_break == 0:
            curr_cost = sess.run(code_space_cost,
                feed_dict = {samples: batch_uniform, x_code: batch_x})
            print 'Cost at iteration ' + str(i+1) + ': ' + str(curr_cost)

        # optimize the moment matching network
        sess.run(code_optimizer, feed_dict = {samples: batch_uniform,
                                              x_code: batch_x})

    # parameters for figure generation
    num_rows = 10; num_cols = 10

    # generate samples from the trained network
    batch_uniform = np.random.uniform(low = -1.0, high = 1.0,
        size = (batch_size, code_space_dims[0]))
    gen_samples   = sess.run(generate, feed_dict = {samples: batch_uniform})

    # generate figure of generated samples
    file_name = dataset + '_code_space.png'
    generateFigure(gen_samples, num_rows, num_cols, image_side, file_name)

parser = argparse.ArgumentParser(description = 'Train GMMN')
parser.add_argument('-d', '--dataset', choices = ['mnist', 'lfw'])
parser.add_argument('-n', '--network', choices = ['data_space', 'code_space'])
args = parser.parse_args()

if args.network == 'data_space':
    trainDataSpaceNetwork(args.dataset)
elif args.network == 'code_space':
    trainCodeSpaceNetwork(args.dataset)
