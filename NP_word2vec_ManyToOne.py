#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import re
import os
import time
from datetime import datetime
from utils import *
import pickle
from gensim.models import word2vec
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# Reading the input Google word2vec model
#model_vec = word2vec.Word2Vec.load_word2vec_format('/Users/ScrmBison/Desktop/GoogleNews-vectors-negative300.bin', binary=True)
model_vec = word2vec.Word2Vec.load_word2vec_format('/Users/Rouzbeh/BoxSync/Fall2016/ESCALES/GoogleNews-vectors-negative300.bin', binary=True)
_VECTOR_SIZE = int(os.environ.get('VECTOR_SIZE', '300'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '500'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.0025'))
_NEPOCH = int(os.environ.get('NEPOCH', '10'))
evaluate_loss_after = 2
vocabulary_size = 5000

_MODEL_FILE = os.environ.get('MODEL_FILE')
vector_size = _VECTOR_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
hidden_dim = _HIDDEN_DIM
learning_rate = _LEARNING_RATE
nepoch = _NEPOCH
print "Reading the input file"
inputfile = open("all_data.label",'r')

inputfile_1 = open("list",'r')
input_1 = inputfile_1.readlines()
freq_voc = []
for item in input_1:
    freq_voc.append(item.split('\n')[0])


label_dic = {}
input_ = inputfile.readlines()
i = 0
y = []
x = []
sent = []
counter = 0
for item in input_:
    # For extracting the label of each sentence
    tag_ = re.findall(r'\w+:',item)
    tag = tag_[0].split(':')[0]
    if tag not in label_dic:
        # If the label is not in my dictionary I will add it to the dictionary
        # label_dic changes labels into numerical values 
        label_dic[tag] = i
        i = i + 1
    # Changing the label into numbers and put them inside y
    y.append(label_dic[tag])
    counter += 1

    sent.append(re.sub(r'\w+:\w+','',item)[0:-2])

sentences = sent
#sentences = itertools.chain(*[nltk.sent_tokenize(pi.lower()) for pi in sent])

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for i,sent in enumerate(sentences)]
# This se
for ii, vall in enumerate(tokenized_sentences):
    for jj,item in enumerate(vall):
        if item in freq_voc:
            tokenized_sentences[ii][jj] = 'stop_words'


# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
# Create the training data
X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])

X_train = X[0:5000]
y_train = y[0:5000]

X_test = X[5000:]
y_test = y[5000:]

class RNNNumpy:
     
    def __init__(self, vector_dim , hidden_dim, label_dim = 6 , bptt_truncate = 6):
        # Assign instance variables
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.vector_dim = vector_dim
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1./vector_dim), np.sqrt(1./vector_dim), (hidden_dim, vector_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (label_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

def forward_propagation(self, x):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, self.hidden_dim))
    s[-1] = np.zeros(self.hidden_dim)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros(self.label_dim)
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        try:
            X_i = model_vec[index_to_word[x[t]]]
        except KeyError:
            X_i = model_vec['unknown']
        s[t] = np.tanh(self.U.dot(X_i) + self.W.dot(s[t-1]))
    o = softmax(self.V.dot(s[-2]))
    return [o, s]
RNNNumpy.forward_propagation = forward_propagation

def predict(self, x):
    # Perform forward propagation and return index of the highest score
    o, s = self.forward_propagation(x)
    return np.argmax(o, axis=0)
 
RNNNumpy.predict = predict


def calculate_total_loss(self, x, y):
    L = 0.0
    # For each sentence...
    for i in np.arange(len(y)):
        o, s = self.forward_propagation(x[i])
        # We only care about our prediction of the "correct" words
        correct_word_predictions = o[y[i]]
        # Add to the loss based on how off we were
        L += -1.0 * (np.log(correct_word_predictions))
    return L
 
def calculate_loss(self, x, y):
    # Divide the total loss by the number of training examples
    # y -> all y_train[i]
    N = len(y)
    return self.calculate_total_loss(x,y)/N
 
RNNNumpy.calculate_total_loss = calculate_total_loss
RNNNumpy.calculate_loss = calculate_loss

def bptt(self, x, y):
    T = len(x)
    # Perform forward propagation
    o, s = self.forward_propagation(x)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(self.U.shape)
    dLdV = np.zeros(self.V.shape)
    dLdW = np.zeros(self.W.shape)
    delta_o = o
    delta_o[y] -= 1.
    # For each output backwards...
    dLdV += np.outer(delta_o, s[T-1].T)
    # Initial delta calculation
    delta_t = self.V.T.dot(delta_o) * (1 - (s[-1] ** 2))
    # Backpropagation through time (for at most self.bptt_truncate steps)
    for bptt_step in np.arange(max(0, (T-1)-self.bptt_truncate), (T-1)+1)[::-1]:
        #print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
        dLdW += np.outer(delta_t, s[bptt_step-1])     
        try:
            X_j = model_vec[index_to_word[x[bptt_step]]]
        except KeyError:
            X_j = model_vec['unknown']
        dLdU += np.outer(delta_t, X_j)   
        # Update delta for next step
        delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
    return [dLdU, dLdV, dLdW]
 
RNNNumpy.bptt = bptt


# Performs one step of SGD.
def numpy_sgd_step(self, x, y, learning_rate):
    # Calculate the gradients
    dLdU, dLdV, dLdW = self.bptt(x, y)
    # Change parameters according to gradients and learning rate
    self.U -= learning_rate * dLdU
    self.V -= learning_rate * dLdV
    self.W -= learning_rate * dLdW
 
RNNNumpy.sgd_step = numpy_sgd_step





# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate, nepoch, evaluate_loss_after):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    print "There is going to be ", nepoch, " epoches"
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5  
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
        # For each training example...
        
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
    filename11 = 'numpy_model.sav'
    pickle.dump(model, open(filename11, 'wb'))


model = RNNNumpy(vector_size,hidden_dim)
train_with_sgd(model, X_train, y_train, learning_rate, nepoch, evaluate_loss_after)



def predict_label(model, X_test,y_test):
    predicted = 0
    correct = 0
    for i in range(len(y_test)):
        predicted = model.predict(X_test[i])
        #print predicted
        if predicted == y_test[i]:
            correct += 1
    accuracy = 100*(correct / float(len(y_test)))
    # next_word_probs = model.forward_propagation(new_sentence)

    return accuracy

filename11 = 'numpy_model.sav'
model = pickle.load(open(filename11, 'rb'))
print "accuracy = %f" % predict_label(model, X_test,y_test), "Test Data"
print "accuracy = %f" % predict_label(model, X_train,y_train), "Train Data"
# tokenized_sentences

print 'nepoch = ',nepoch ,'vocabulary_size = ' ,vocabulary_size ,'hidden_dim = ' ,hidden_dim ,'learning_rate = ',learning_rate





