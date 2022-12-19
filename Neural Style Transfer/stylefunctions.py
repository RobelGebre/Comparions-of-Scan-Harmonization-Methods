from __future__ import print_function
import os
import IPython.display as display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras
from keras.models import Model
from keras.layers import Activation
from keras.utils.vis_utils import plot_model
from numba import cuda, jit, njit
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import random
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from nilearn import plotting

import SimpleITK as sitk
import sys
import scipy.ndimage

model_path = 'ReseNet-18.h5' #change here

def model_layers(layer_names):

    model_custom = keras.models.load_model(model_path)
    model_custom.trainable = False

    outputs = [model_custom.get_layer(name).output for name in layer_names]

    model_custom = tf.keras.Model([model_custom.input], outputs)
    
    return model_custom

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijck,bijdk->bcdk', input_tensor, input_tensor) #modified to accomodate 3D format
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[0]*input_shape[1]*input_shape[2], tf.float32) 
    return result/(num_locations)   


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.model_layers = model_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.model_layers.trainable = True   

    def call(self, inputs):
        "Expects float input in [0,1]"
 
        preprocessed_input  = tf.convert_to_tensor(inputs)
        preprocessed_input = preprocessed_input*255.0 
        
        outputs = self.model_layers(preprocessed_input)
               
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        
        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
    

initial_learning_rate = 0.001 #-1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate,
                                                          decay_steps=100000000,
                                                          decay_rate=0.9, #
                                                          staircase=True,
                                                            )

opt = tf.keras.optimizers.Adagrad(
    learning_rate=initial_learning_rate, initial_accumulator_value=1e-1, epsilon=1e-5,
    name='Adagrad') 

style_weight=1e-3 
content_weight=style_weight*1e3 #beta = 1000 x alpha, according to the Gatys et al.

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
    
def high_pass_x_y(image):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    
    z_var = image[1:, :, :, :] - image[:-1, :, :, :]
    
    
    x_var = x_var.astype('float32')
    y_var = y_var.astype('float32')
    
    z_var = z_var.astype('float32')
    return x_var, y_var, z_var

def total_variation_loss(image):
    x_deltas, y_deltas, z_deltas = high_pass_x_y(image)
    tvl = tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas)) + tf.reduce_sum(tf.abs(z_deltas))
    return tvl