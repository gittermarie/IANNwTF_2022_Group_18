import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

'''
This file contains 5 different models to test out different optimization techniques

- model from last week
- model with batch normalization
- model with dropout rate
- model with regularization technique (L2 loss)
- model with Glorot Normal kernel initializer
'''

# basic model class that all other models inherit from
# it has all the shared attributes 
# individual Models are children that inherit from basic model and have added optimization-features
class Basic_CNN(tf.keras.Model):
    def __init__(self):
        
        super().__init__()

        # optimzer, metrics, loss
        self.optimizer = tf.keras.optimizers.Adam()

        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc") 
                       ]

        self.loss_function = tf.keras.losses.CategoricalCrossentropy()
        
        # list of all layers to iterate through during call
        self.layer_list = []
        
    # call funtion
    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
    
    # metrics property
    @property
    def metrics(self):
        return self.metrics_list
        # return a list with all metrics in the model

    # reset all metrics objects
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()
    
    # train step method
    @tf.function
    def train_step(self, data):
        
        x, targets = data
        
        # calculate and backpropagate gradients
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # update accuracy
        for metric in self.metrics[1:]:
            metric.update_state(targets,predictions)

        # Return a dictionary mapping metric names to current value to keep track of training
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        
        # same as in training but without backpropagating
        x, targets = data
        predictions = self(x, training=False)
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        # update loss metric
        self.metrics[0].update_state(loss)
        # update accuracy metric
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}

# Last weeks CNN for comparison
class CNN(Basic_CNN):
    
    def __init__(self):  
        # inherit functionalities from "Basic_CNN" parent class
        super().__init__()
        
        # layers
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())

        self.layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))

# CNN with batchnormalization
class CNN_batchnorm(Basic_CNN):
    
    def __init__(self):
        # not sure about this init 
        super().__init__()
        
        # layers
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())
        
        # batch normalization
        self.layer_list.append(tf.keras.layers.BatchNormalization()) # training should be autom. set to the right value?!

        self.layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))
    
# CNN with dropout layers   
class CNN_dropout(Basic_CNN):
    
    def __init__(self):
        
        super().__init__()   
        
        # layers
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        # dropout layer
        self.layer_list.append(tf.keras.layers.Dropout(0.2))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        # dropout layer
        self.layer_list.append(tf.keras.layers.Dropout(0.1))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())

        self.layer_list.append(tf.keras.layers.Dense(10, activation='softmax'))

# CNN with regularization technique (L2 loss)    
class CNN_L2loss(Basic_CNN):
    
    def __init__(self):
        
        super().__init__()   
        
        # layers
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())

        self.layer_list.append(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer='l2')) #l2 loss added 
    
# model with Glorot Normal kernel-initialization instead of Uniform 
class CNN_GlorotNormal(Basic_CNN):
    
    def __init__(self):
        
        super().__init__()
        
        # GlorotNormal Initializer
        initializer = tf.keras.initializers.GlorotNormal()
        
        # layers
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, kernel_initializer=initializer, 
                                                      padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=24, kernel_size=3, kernel_initializer=initializer, 
                                                      padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        
        
        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, kernel_initializer=initializer, 
                                                      padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.Conv2D(filters=48, kernel_size=3, kernel_initializer=initializer, 
                                                      padding='same', activation='relu'))
        self.layer_list.append(tf.keras.layers.GlobalAvgPool2D())

        
        self.layer_list.append(tf.keras.layers.Dense(10, activation='softmax', 
                                                     kernel_initializer=initializer))