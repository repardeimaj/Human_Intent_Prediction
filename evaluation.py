#! /usr/bin/env/ python3

#****************************************************************************#

#                             evaluation.py                                  #

# file for testing and evaluating the trained models on the long sequences   #

#****************************************************************************#


import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


import math
import pandas as pd
import io

import sklearn
from sklearn.metrics import classification_report


import visualizer as viz

from sequence_model import sequence_model

class evaluation:

    sequences = []
    model = None
    labels = []
    seq = None

    def __init__(self):
        self.start()

    def start(self):
        #use the sequence class for preparing the data for input to the model
        self.seq = sequence_model()
        self.sequences = self.seq.get_sequences()
        max_len = self.seq.get_max_length()
        self.labels = self.seq.get_sequence_labels()

        # for i in range(len(self.sequences)):
        #     viz.visualize_sequence(self.sequences[i], "sequence " + str(i), 1)

        #the sequence to test on, see sequence #.mp4
        #note which ones were used in training, and which can be used for evaluation
        idx = 0

        #self.model = self.import_model("/home/paternaincomputer/HIP_ws/RNNModel_low_loss")
        self.import_model("/home/paternaincomputer/HIP_ws/GNN_Model_len_20")
        #self.import_model("/home/paternaincomputer/HIP_ws/CNN_Model_len_20")

        print(self.model.summary())
        self.sequences[idx] = self.seq.pad_sequence(self.sequences[idx], len(self.sequences[idx]) + max_len)

        #pred_labels, probs = self.evaluate(self.model,self.distort_data(self.sequences[idx]), max_len)
        pred_labels, probs = self.evaluateGNN(self.model,self.distort_data(self.sequences[idx]), max_len)

        temp = 0
        for i in range(len(pred_labels)):
            if math.isclose(pred_labels[i], self.labels[idx][i], rel_tol=1e-1, abs_tol=0.0):
                temp += 1
            
        #absolute accuracy
        print("ACCURACY:")
        acc = float(temp / len(pred_labels))
        print(acc)

        #pred_labels = tf.concat([np.zeros(max_len),pred_labels], axis=0)

        plt.figure(figsize=(12, 4))
        plt.plot(pred_labels, label='Prediction')
        plt.plot(self.labels[idx], label='Ground Truth')
        plt.xlabel('Timestep')
        plt.ylabel('State')
        plt.legend()
        plt.show()

    #evaluate the model on a testing sequence 
    def evaluate(self, model,  sequence, length):
        predicted_labels = []
        predicted_probabilities = []

        for i in range(length, len(sequence)):
            temp = np.array([sequence[i-length:i]])
            predicted_prob = model.predict(temp)
            predicted_probabilities.append(predicted_prob)
            #add a slight offset so plot lines don't overlap
            predicted_labels.append(np.argmax(predicted_prob) + 0.05)
        
        return predicted_labels, predicted_probabilities
    
    #preprocess the graph convolution aggregations, then evaluate the model
    def evaluateGNN(self, model,  sequence, length):
        predicted_labels = []
        predicted_probabilities = []

        A = viz.adj(viz.generateGraph(sequence[0][0],sequence[0][1],sequence[0][2]))
        print(tf.shape(A))

        joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        hops = 3*np.ones((32), dtype=np.int32)


        for i in range(length, len(sequence)):
            temp = np.array(sequence[i-length:i])
            print(tf.shape(temp))
            temp = np.array([self.seq.GraphConv(temp,joints,hops,A)])
            predicted_prob = model.predict(temp)
            predicted_probabilities.append(predicted_prob)
            predicted_labels.append(np.argmax(predicted_prob) + 0.05)
        
        return predicted_labels, predicted_probabilities


    #distort the data by turning some joints to 0
    def distort_data(self, sequence):
        sequence2 = sequence.numpy().tolist()
        for i in range(len(sequence)):
            for k in range(7):
                sequence2[i][k][6] = 0
                sequence2[i][k][12] = 0
                sequence2[i][k][4] = 0
                sequence2[i][k][15] = 0
                sequence2[i][k][25] = 0
                #sequence2[i][k][26] = 0
                sequence2[i][k][30] = 0
                sequence2[i][k][22] = 0
                #sequence2[i][k][28] = 0
                #sequence2[i][k][9] = 0

        return np.array(sequence2)


    def import_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

if __name__ == "__main__":
    _ = evaluation()
