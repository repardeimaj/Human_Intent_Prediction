import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class GCNLayer(Layer):
    def __init__(self, filters, hops, joints, A, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.filters = filters
        self.filters = filters
        # number of hop neighbors
        self.hops = hops
        #joints to take info from
        self.joints = joints
        self.A = A 


    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.kernel = self.add_weight("kernel", shape=(self.filters, self.hops,20), 
                                                        initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        inputs = tf.cast(inputs, dtype=tf.float32)
        output = self.GraphConv(inputs, self.joints, self.hops, self.A)
        return output
  
    def GraphConv(self, x, joints, hops, A):
        sequence = []
        A = tf.tile(tf.expand_dims(A, axis=0), [tf.shape(x)[0], 1, 1])
        for l in range(len(x)):
            outputs = []

            for n in range(self.filters):
                conv = 0
                for k in range(hops):
                 
                    weighted_features = tf.matmul(tf.pow(A,k), x[:,l,:,:])
                    conv += self.kernel[n][k][l]*weighted_features
                 
                
                outputs.append(conv)
            sequence.append(tf.math.reduce_sum(outputs, axis=0))
        return sequence
    
    def matrix_power(self, A, k):
        result = A
        for _ in range(k - 1):
            result = tf.matmul(result, A)
        return tf.cast(result, dtype=tf.float32)
    
