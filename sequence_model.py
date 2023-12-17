#! /usr/bin/env/ python3

#****************************************************************************#

#                       sequence_model.py                                    #

#    file for training the sequence models for human intent prediction       #

#****************************************************************************#

import tensorflow as tf
from tensorflow import keras
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt

import random
import math
import pandas as pd
import io

import bagpy
from bagpy import bagreader
import rosbag
from visualization_msgs.msg import MarkerArray

import yaml

import sklearn
from sklearn.metrics import classification_report

import os
from os import walk

import visualizer as viz
from keras import backend as K

from GNC_Layer import GCNLayer

import spektral

import tensorflow_gnn as tfgnn

class sequence_model:


    labels = ["crouching", "follow", "meet", "lifting", "idle"]

    training_data = []
    validation_data = []
    training_labels = []
    validation_labels = []

    model = None

    max_length = 0
    
    padded_data = []
    data_labels = []

    sequences = []
    sequence_labels = []

    def __init__(self):
        self.start()
    
    #run once on initialization. Setup all data, read rosbags, etc.
    def start(self):

        data, labels, self.sequences = self.retrieve_data()
        self.sequence_labels = self.get_sequence_labels()

        #window length
        self.max_length = 20

        #add data from long recorded sequences to training data
        seq_data, seq_labels = self.data_from_sequences([1,4], self.max_length)

        data += seq_data
        labels += seq_labels

        self.padded_data, self.data_labels = self.shape_data(data, labels, self.max_length)

    #Prepare the data and train the model (can be re-run to re-shuffle data before training again)
    def organize_data(self):
        #partition and shuffle the data
        self.training_data, self.training_labels, self.validation_data, self.validation_labels = self.partition_data(self.padded_data, self.data_labels, 0.75)

        #adjacency matrix
        A = viz.adj(viz.generateGraph(self.training_data[0][0][0],self.training_data[0][0][1],self.training_data[0][0][2]))

        #RNN only
        # self.model = self.model_setup_no_GNN(self.max_length)

        #CNN   
        #self.model = self.model_setup_CNN(self.max_length)
        #self.train_model_for(400)

        self.model = self.test_model(32,3,20)

        #to best compare with CNN, agregate over every single joint
        joints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

        #3 hops for every joint
        hops = 3*np.ones((32), dtype=np.int32)

        #preprocess graph convolutions for every sequence of training data
        temp = []
        for i in range(len(self.training_data)):
            temp.append(self.GraphConv(self.training_data[i], joints, hops,A))
        self.training_data = np.array(temp)

        temp = []
        for i in range(len(self.validation_data)):
            temp.append(self.GraphConv(self.validation_data[i], joints, hops,A))
        self.validation_data = np.array(temp)

        self.train_model_to(0.2, 0.98, 0.4, 0.96, 800)

        self.model.save("/home/paternaincomputer/HIP_ws/Models/GNN_Model_len_" + str(self.max_length))

    #train the model for some # epochs
    def train_model_for(self, epochs):

        K.set_value(self.model.optimizer.learning_rate, 0.000075)

        # Lists to store training metrics
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        # Training loop
        for epoch in range(epochs):
            # Perform training steps
            history = self.model.fit(self.training_data, self.training_labels, validation_data=(self.validation_data,  self.validation_labels))

            # Record training metrics
            train_loss.append(history.history['loss'])
            train_acc.append(history.history['accuracy'])
            val_loss.append(history.history['val_loss'])
            val_acc.append(history.history['val_accuracy'])

       

        # Disable interactive mode
        plt.ioff()

        # Plot the training metrics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()        

        predictions = self.model.predict(self.validation_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # Generate classification report
        report = classification_report(self.validation_labels, predicted_labels)
        print(predicted_labels)
        print(report)

    #train the model until desired training and validation losses and accuracies are reached
    def train_model_to(self, t_l, t_a, v_l, v_a, max_epochs):

        K.set_value(self.model.optimizer.learning_rate, 0.0001)

        # Lists to store training metrics
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        end = False
        n = 0
        # Training loop
        while (not end):
                    n += 1
                    #Perform training steps
                    history = self.model.fit(self.training_data, self.training_labels, validation_data=(self.validation_data,  self.validation_labels))

                    # Record training metrics
                    train_loss.append(history.history['loss'])
                    train_acc.append(history.history['accuracy'])
                    val_loss.append(history.history['val_loss'])
                    val_acc.append(history.history['val_accuracy'])

                    accuracy = val_acc[len(val_acc)-1][0]
                    if (train_loss[len(train_loss)-1][0] < t_l and train_acc[len(train_acc)-1][0] > t_a \
                        and val_loss[len(val_loss)-1][0] < v_l and val_acc[len(val_acc)-1][0] > v_a):
                        end = True
                    
                    if n > max_epochs:
                        self.organize_data()
                        return
                    
       

        # Disable interactive mode
        plt.ioff()

        # Plot the training metrics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()        

        predictions = self.model.predict(self.validation_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # Generate classification report
        report = classification_report(self.validation_labels, predicted_labels)
        print(predicted_labels)
        print(report)
 
    #can be used to get longest sequence length when using longer window sizes
    def max_sequence_length(self, data):
        lengths = [len(data[i]) for i in range(len(data))]
        return max(lengths)

    #get the data from the rosbags
    def retrieve_data(self):
        #Look through file system and get the bag file names in each folder
        folders = ["crouching", "follow", "meet", "lifting", "idle", "sequence"]
        base_path = "/home/paternaincomputer/HIP_ws/Data/bag/"
        points = []
        labels = []
        sequences = []
        for i in range(len(folders)):
            
            for (dirpath, dirnames, filenames) in walk(base_path + folders[i]):
                
                for filename in filenames:
                    try:
                        if i > 4:
                            sequences.append(self.process_bag(base_path + "/" + folders[i]+ "/" + filename))
                        else:

                            sequence = self.process_bag(base_path + "/" + folders[i]+ "/" + filename)

                            points.append(sequence)
                            labels.append(i)
                    except Exception as e:
                        print("failed inmport: " + filename +": " +folders[i])
                        print(e)

                
                break

        return points, labels, sequences

    #zero-pad the data from the beginning of the sequence to reach a desired length
    def pad_data(self, data, max_length,labels):
        padded = []
        output_labels = []
        for i in range(len(data)):
            try:
                if len(data[i]) < max_length:
                    
                    padding = np.zeros((max_length - len(data[i]), 7,32),dtype=np.float32)
                    temp = [padding, data[i]]
                    padded.append(tf.concat(temp, axis=0))
                    output_labels.append(labels[i])

                else:
                    padded.append(data[i])
                    output_labels.append(labels[i])
            except Exception as e:
                print("failed: ")
                print(self.labels[labels[i]])
                


        return padded, output_labels
    
    #zero-pad a single sequence
    def pad_sequence(self, sequence, max_length):
        padding = np.zeros((max_length - len(sequence), 7,32),dtype=np.float32)
        temp = [padding, sequence]
        return(tf.concat(temp, axis=0))
    
    #get the data from a single rosbag given the file path
    def process_bag(self, file_path):
        sequence = []
        bag = rosbag.Bag(file_path)
        for topic,msg,t in bag.read_messages():
            markers = msg.markers
            joints = []
            
            for i in range(32):
                temp = []
            
                try:
                    temp.append(markers[i].pose.position.x)
                    temp.append(markers[i].pose.position.z)
                    temp.append(-markers[i].pose.position.y)
                    temp.append(markers[i].pose.orientation.x)
                    temp.append(markers[i].pose.orientation.y)
                    temp.append(markers[i].pose.orientation.z)
                    temp.append(markers[i].pose.orientation.w)
                    joints.append(temp)
                except:
                    joints.append(np.zeros(7))
           
            joints = np.transpose(joints)

            sequence.append(joints)
        bag.close()
        return sequence

    #break a sequence into some 'factor' number of evenly sampled sequences
    def subsample(self, data, labels, factor):

        new_data = []
        new_labels = []
        for i in range(len(data)):
            j = 0
            temp = []
            for _ in range(factor):
                temp.append([])

            while j < int(len(data[i]) / factor) * factor:
                for k in range(factor):
                    temp[k].append(data[i][j + k])

                j += factor
        
            for l in range(factor):
                new_data.append(temp[l])
                new_labels.append(labels[i])

        return new_data, new_labels

    #define an RNN model 
    def model_setup_no_GNN(self, max):
        model = tf.keras.Sequential([
        tf.keras.layers.Reshape(( max, 32 * 7), input_shape=(max, 7, 32)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.SimpleRNN(units=75, activation="tanh"),#kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.Dense(5, activation="softmax")
        ])

        print(model.summary())

        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

        return model
    
    #define a CNN model for the data
    def model_setup_CNN(self, max):

        model = tf.keras.Sequential([
        
        tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), activation='relu', input_shape=(max, 7, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),

        tf.keras.layers.Reshape((max, -1)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.13)),
        tf.keras.layers.LSTM(units=15, activation="tanh"),#kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        #tf.keras.layers.Dropoutestt(0.25),
        #tf.keras.layers.Dense(20, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(5, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

        print(model.summary())


        return model
    
    #define a GNN model for the data
    def model_setup_GNN(self, joints, hops, length):
        #input preprocessed aggregations
        input = tf.keras.layers.Input(shape=(length,joints,hops,7))
        reshaped = tf.keras.layers.Reshape((length,hops,-1))(input)

        #perform element wise multiplication of the conv filters and the preprocessed aggregations for each timestep
        time_dist = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=32, kernel_size=hops, activation='relu', padding='valid'))(reshaped)
        reshaped2 = tf.keras.layers.Reshape((20,32))(time_dist)

        #convolution layer over time dimension
        conv = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.25))(reshaped2)
        lstm = tf.keras.layers.LSTM(units=15, activation="tanh")(conv)

        dense = tf.keras.layers.Dense(5, activation='softmax')(lstm)

        model = tf.keras.Model(inputs=input, outputs=dense)
        
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


        print(model.summary())

        return model

    #label the long testing sequences
    def get_sequence_labels(self):
        labels = []
        #["0crouching", "1follow", "2meet", "3lifting", "4idle"]
        # Meet follow crouching lifting
        temp = np.zeros(len(self.sequences[0]))
        label_ranges = [(0,111,2),(111,325,1), (325,364,0), (364,396,3)]
        for start, end, label in label_ranges:
            temp[start:end] = label
        labels.append(temp)
        # Meet follow crouching lifting variation
        temp = np.zeros(len(self.sequences[1]))
        label_ranges = [(0,167,2),(167,291,1), (291,323,0), (323,378,3)]
        for start, end, label in label_ranges:
            temp[start:end] = label
        labels.append(temp)
        # Meet follow crouching lifting idle meet idle follow crouching 
        temp = np.zeros(len(self.sequences[2]))
        label_ranges = [(0,161,2),(161,262,1), (262,288,0), (288,403,3),(403,542,4),(542,644,2),(644,753,4),(753,814,1),(814,846,0)]
        for start, end, label in label_ranges:
            temp[start:end] = label
        labels.append(temp)
        # Meet follow idle crouching lifting
        temp = np.zeros(len(self.sequences[3]))
        label_ranges = [(0,144,2),(144,252,1), (252,283,0), (283,304,3), (304,400,4), (400, 438, 0), (438, 483, 3)]
        for start, end, label in label_ranges:
            temp[start:end] = label
        labels.append(temp)
        # Meet follow idle crouching lifting variation
        
        # Meet follow crouching lifting meet follow
        # Meet follow crouching lifting meet follow variation
        # Meet follow crouching lifting idle crouching lifting

        temp = np.zeros(len(self.sequences[4]))
        label_ranges = [(0,106,2),(106,208,1),(208,232,4), (232,261,0), (283,375,3)]
        for start, end, label in label_ranges:
            temp[start:end] = label
        labels.append(temp)
        # Meet follow crouching lifting Meet follow crouching lifting
        # Meet follow idle crouching lifting idle Meet follow crouching lifting
        temp = np.zeros(len(self.sequences[5]))
        label_ranges = [(0,100,2), (100, 222, 1), (222, 307, 4), (307, 347, 0), (307, 396, 3), (396, 552, 4), (552, 665, 2), (665, 768, 1), (768, 792, 0), (792,851, 3)]
        for start, end, label in label_ranges:
            temp[start:end] = label
        labels.append(temp)
        #["0crouching", "1follow", "2meet", "3lifting", "4idle"]

        return labels

    def shape_data(self, data, labels, length):
        shaped_labels = []
        shaped_data = []

        for i in range(len(data)):
            #print(len(data[i]))


            if len(data[i]) == 0:
                ()
            elif len(data[i]) < length:

                shaped_data.append(self.pad_sequence(data[i], length))
                shaped_labels.append(labels[i])
            elif len(data[i]) == length:
                shaped_data.append(data[i])
                shaped_labels.append(labels[i])
            else:
                for j in range(1):
                    start = random.randrange(0, len(data[i]) - length - 1)
                    end = start + length
                    shaped_data.append(data[i][start:end])
                    shaped_labels.append(labels[i])
        
        return shaped_data, shaped_labels

    #return max length for external access  
    def get_max_length(self):
        return self.max_length

    #return sequences for external access
    def get_sequences(self):
        return self.sequences
    
    #shuffle and partition data for specified training/validation split
    def partition_data(self, datapoints, datalabels, training_val_ratio):
        datapoints = np.array(datapoints)
        datapoints = datapoints.astype(np.float32)
        datalabels = np.array(datalabels)

        shuffledpoints, shuffledlabels = sklearn.utils.shuffle(datapoints, datalabels)
        n = len(datapoints)
        idx = math.ceil(n * training_val_ratio)

        training_data = shuffledpoints[0:idx]
        training_labels = shuffledlabels[0:idx]
        validation_data = shuffledpoints[idx+1:n]
        validation_labels = shuffledlabels[idx+1:n]

        return training_data, training_labels, validation_data, validation_labels
    
    #get training data from a specific range of the long sequences
    def data_from_sequences(self, Range, length):
        data = []
        labels = []

        for i in Range:
            padded = self.pad_sequence(self.sequences[i],len(self.sequences[i]) + length)

            for j in range(length, len(self.sequences[i]) - 1):
                data.append(self.sequences[i][j-length:j])
                labels.append(self.sequence_labels[i][j - length])

        return data, labels

    #preprocess the graph convolution aggreations for a sequence of feature matrixes and an adjacency matrix S
    def GraphConv(self, x, joints, hops,S):
        #agregate for some number of hops
        #over specified joints
        seq = []
        for j in range(len(x)):
            temp = []
            for i in range(len(joints)):
                conv = []

                for k in range(hops[i]):
                    conv.append((np.linalg.matrix_power(S,k) @ np.transpose(x[j]))[joints[i]])
                temp.append(conv)
            seq.append(temp)
        return seq

        

if __name__ == "__main__":
    random.seed()
    model = sequence_model()
    model.organize_data()
