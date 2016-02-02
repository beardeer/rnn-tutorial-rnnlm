#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split

from data_process import dkt_input_data_reader


x, y, v_size = dkt_input_data_reader()

print v_size

_VOCABULARY_SIZE = v_size
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '200'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

X_data = np.asarray(x)
y_data = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
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
            # ADDED! Saving model oarameters
            # save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

        all_lables = []
        all_predictions = []
        for n in range(len(y_test)):
            x_item = X_test[n]
            y_item = y_test[n]

            if len(x_item) <= 2:
                continue

            model_out = model.predict(x_item)
            
            lables = []
            predictions = []

            off_set = (v_size - 2) / 2 # 111

            for m in range(len(model_out) - 1):
                y_value = model_out[m]
                true_value = y_item[m]

                if true_value >= off_set + 2:
                    predictions.append(y_value[true_value])
                    lables.append(1.0)
                else:
                    predictions.append(y_value[true_value + off_set])
                    lables.append(0.0)

            all_lables.extend(lables)
            all_predictions.extend(predictions)
        
        print 'auc ...'
        print roc_auc_score(all_lables, all_predictions)


vocabulary_size = _VOCABULARY_SIZE


model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
t1 = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
t2 = time.time()
print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

# if _MODEL_FILE != None:
#     load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)