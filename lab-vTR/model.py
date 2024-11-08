import numpy as np
from numpy import newaxis
import pandas as pd
import sys
import math
import os
import json
import csv
import pandas
import random
import scipy
import matplotlib.pyplot as plt

from data_preprocess import preprocess
from scipy.stats import spearmanr, pearsonr

from hyperopt import fmin
from hyperopt import hp
from hyperopt import tpe

# specifically for model visualization
import keras
from keras.utils.vis_utils import model_to_dot
from keras.utils.vis_utils import plot_model
from keras.models import Model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D, BatchNormalization, Activation, concatenate, ReLU, Add
from tensorflow.keras import backend as K, regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda
from tensorflow import keras

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.utils import shuffle

#Reproducibility
# seed = random.randint(1, 1000)
seed = 70

np.random.seed(seed)
tf.random.set_seed(seed)


class ConvolutionLayer(Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 data_format,
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 __name__='ConvolutionLayer',
                 **kwargs):
        super(ConvolutionLayer, self).__init__(filters=filters,
                                               kernel_size=kernel_size,
                                               activation=activation,
                                               use_bias=use_bias,
                                               kernel_initializer=kernel_initializer,
                                               **kwargs)
        self.run_value = 1

    def call(self, inputs):

      ## shape of self.kernel is (12, 4, 512)
      ##the type of self.kernel is <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>

        print("self.run value is", self.run_value)
        if self.run_value > 2:

            x_tf = self.kernel  # x_tf after reshaping is a tensor and not a weight variable :(
            x_tf = tf.transpose(x_tf, [2, 0, 1])

            alpha = 1000
            beta = 1/alpha
            bkg = tf.constant([0.25, 0.25, 0.25, 0.25])
            bkg_tf = tf.cast(bkg, tf.float32)
            filt_list = tf.map_fn(lambda x: tf.math.scalar_mul(beta, tf.subtract(tf.subtract(tf.subtract(tf.math.scalar_mul(alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis=1), axis=1)), tf.expand_dims(tf.math.log(tf.math.reduce_sum(
                tf.math.exp(tf.subtract(tf.math.scalar_mul(alpha, x), tf.expand_dims(tf.math.reduce_max(tf.math.scalar_mul(alpha, x), axis=1), axis=1))), axis=1)), axis=1)), tf.math.log(tf.reshape(tf.tile(bkg_tf, [tf.shape(x)[0]]), [tf.shape(x)[0], tf.shape(bkg_tf)[0]])))), x_tf)

            # def dummy(x):
            #     return
            #     tf.math.scalar_mul(
            #         beta,
            #         tf.subtract(
            #             tf.subtract(
            #                 tf.subtract(
            #                     tf.math.scalar_mul(alpha, x),
            #                     tf.expand_dims(
            #                         tf.math.reduce_max(
            #                             tf.math.scalar_mul(alpha, x),
            #                             axis=1
            #                         ),
            #                         axis=1
            #                     )
            #                 ),
            #                 tf.expand_dims(
            #                     tf.math.log(
            #                         tf.math.reduce_sum(
            #                             tf.math.exp(
            #                                 tf.subtract(
            #                                     tf.math.scalar_mul(alpha, x),
            #                                     tf.expand_dims(
            #                                         tf.math.reduce_max(
            #                                             tf.math.scalar_mul(alpha, x),
            #                                             axis=1
            #                                         ),
            #                                         axis=1
            #                                     )
            #                                 )
            #                             ),
            #                             axis=1
            #                         )
            #                     ),
            #                     axis=1
            #                 )
            #             ),
            #             tf.math.log(
            #                 tf.reshape(
            #                     tf.tile(
            #                         bkg_tf,
            #                         [ tf.shape(x)[0] ]
            #                     ),
            #                     [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ]
            #                 )
            #             )
            #         )
            #     )


            # def calculate(x):
            #     alpha_x = tf.math.scalar_mul(alpha, x)
            #     ax_reduced = tf.math.reduce_max(alpha_x, axis=1)
            #     axr_expanded = tf.expand_dims(ax_reduced, axis=1)
            #     ax_sub_axre = tf.subtract(alpha_x, axr_expanded)
            #     softmaxed = tf.math.reduce_sum(tf.math.exp(ax_sub_axre), axis=1)
            #     sm_log_expanded = tf.expand_dims(tf.math.log(softmaxed), axis=1)
            #     axsaxre_sub_smle = tf.subtract(ax_sub_axre, sm_log_expanded)

            #     bkg_streched = tf.tile(bkg_tf, [ tf.shape(x)[0] ])
            #     bkg_stacked = tf.reshape(bkg_streched, [ tf.shape(x)[0], tf.shape(bkg_tf)[0] ])
            #     bkgs_log = tf.math.log(bkg_stacked)

            #     return tf.math.scalar_mul(beta, tf.subtract(axsaxre_sub_smle, bkgs_log))


            # filt_list = tf.map_fn(calculate, x_tf)




            #print("type of output from map_fn is", type(filt_list)) ##type of output from map_fn is <class 'tensorflow.python.framework.ops.Tensor'>   shape of output from map_fn is (10, 12, 4)
            #print("shape of output from map_fn is", filt_list.shape)
            #transf = tf.reshape(filt_list, [12, 4, self.filters]) ##12, 4, 512
            transf = tf.transpose(filt_list, [1, 2, 0])
            ##type of transf is <class 'tensorflow.python.framework.ops.Tensor'>
            # type of outputs is <class 'tensorflow.python.framework.ops.Tensor'>
            outputs = self._convolution_op(inputs, transf)

        else:
            outputs = self._convolution_op(inputs, self.kernel)

        self.run_value += 1
        return outputs


class nn_model:
    def __init__(self, fasta_file, readout_file, filters, kernel_size, pool_type, regularizer, activation_type, epochs, batch_size):
        """initialize basic parameters"""
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_type = pool_type
        self.regularizer = regularizer
        self.activation_type = activation_type
        self.epochs = epochs
        self.batch_size = batch_size
        #self.stride = stride
        #self.create_model()
        self.fasta_file = fasta_file
        self.readout_file = readout_file

        self.eval()
        #self.filter_importance()
        #self.cross_val()
        #self.hyperopt_tuner()

    def create_model(self):
        # different metric functions
        def coeff_determination(y_true, y_pred):
            SS_res = K.sum(K.square(y_true-y_pred))
            SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
            return (1 - SS_res/(SS_tot + K.epsilon()))

        def auroc(y_true, y_pred):
            return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

        # building model
        prep = preprocess(self.fasta_file, self.readout_file)
        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()
        # if want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        readout = dict["readout"]
        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]

        dim_num = fw_fasta.shape

        # To build this model with the functional API,
        # you would start by creating an input node:
        forward = keras.Input(shape=(dim_num[1], dim_num[2]), name='forward')
        reverse = keras.Input(shape=(dim_num[1], dim_num[2]), name='reverse')

        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', input_shape=(dim_num[1],dim_num[2]), use_bias = False)
        ## with trainable = False
        #first_layer = Conv1D(filters=self.filters, kernel_size=self.kernel_size, kernel_initializer = my_init, data_format='channels_last', input_shape=(dim_num[1],dim_num[2]), use_bias = False, trainable=False)
        first_layer = ConvolutionLayer(
            filters=self.filters, kernel_size=self.kernel_size, data_format='channels_last', use_bias=True)

        fw = first_layer(forward)
        bw = first_layer(reverse)

        concat = concatenate([fw, bw], axis=1)

        pool_size_input = concat.shape[1]

        #concat = ReLU()(concat)
        #concat = Dense(1, activation= 'sigmoid')(concat)

        if self.pool_type == 'Max':
            pool_layer = MaxPooling1D(pool_size=pool_size_input)(concat)
        elif self.pool_type == 'Ave':
            pool_layer = AveragePooling1D(pool_size=pool_size_input)(concat)
        elif self.pool_type == 'custom':
            def out_shape(input_shape):
                shape = list(input_shape)
                print(input_shape)
                shape[0] = 10
                return tuple(shape)
            #model.add(Lambda(top_k, arguments={'k': 10}))

            def top_k(inputs, k):
                # tf.nn.top_k Finds values and indices of the k largest entries for the last dimension
                print(inputs.shape)
                inputs2 = tf.transpose(inputs, [0, 2, 1])
                new_vals = tf.nn.top_k(inputs2, k=k, sorted=True).values
                # transform back to (None, 10, 512)
                return tf.transpose(new_vals, [0, 2, 1])

            pool_layer = Lambda(top_k, arguments={'k': 2})(concat_relu)
            pool_layer = AveragePooling1D(pool_size=2)(pool_layer)
        elif self.pool_type == 'custom_sum':
            ## apply relu function before custom_sum functions
            def summed_up(inputs):
                #nonzero_vals = tf.keras.backend.relu(inputs)
                new_vals = tf.math.reduce_sum(inputs, axis=1, keepdims=True)
                return new_vals
            pool_layer = Lambda(summed_up)(concat_relu)
        else:
            raise NameError('Set the pooling layer name correctly')

        flat = Flatten()(pool_layer)

        after_flat = Dense(32)(flat)

        # Binary classification with 2 output neurons
        if self.regularizer == 'L_1':
            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            ## trainable = False with learned bias

            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(after_flat)
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l1(
                0.001), activation='sigmoid')(after_flat)
        elif self.regularizer == 'L_2':
            #outputs = Dense(1, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.001), activation= self.activation_type)(flat)
            ## trainable = False with learned bias
            outputs = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(
                0.001), activation=self.activation_type)(after_flat)
        else:
            raise NameError('Set the regularizer name correctly')

        #weight_forwardin_0=model.layers[0].get_weights()[0]
        #print(weight_forwardin_0)
        model = keras.Model(inputs=[forward, reverse], outputs=outputs)

        #print model summary
        model.summary()

        #model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auroc])

        return model

    def filter_importance(self):
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()

        # if want dinucleotide sequences
        # dict = prep.dinucleotide_encode()

        # print maximum length without truncation
        np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        # seed = random.randint(1, 1000)

        x1_train, x1_test, y1_train, y1_test = train_test_split(
            fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            rc_fasta, readout, test_size=0.1, random_state=seed)
        #assert x1_test == x2_test
        #assert y1_test == y2_test

        model = self.create_model()

        # change from list to numpy array
        y1_train = np.asarray(y1_train)
        y1_test = np.asarray(y1_test)
        y2_train = np.asarray(y2_train)
        y2_test = np.asarray(y2_test)

        # Copy the original target values for later uses
        y1_train_orig = y1_train.copy()
        y1_test_orig = y1_test.copy()

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        ## Change it to categorical values
        y1_train = keras.utils.to_categorical(y1_train, 2)
        y1_test = keras.utils.to_categorical(y1_test, 2)

        # Restore the weights
        #weight_dir = './data/E13RACtrlF1_E13RAMutF1_DMR_toppos2000/checkpoint/my_checkpoint'
        weight_dir = '/Users/minjunpark/Documents/MuSeAM/classification/saved_weights/my_checkpoint'

        model.load_weights(weight_dir)

        #######*******************************
        pred_train = model.predict({'forward': x1_train, 'reverse': x2_train})

        # See which label has the highest confidence value
        predictions_train = np.argmax(pred_train, axis=1)

        print(y1_train_orig[0:10])
        print(predictions_train[0:10])

        true_pred = 0
        false_pred = 0
        for count, value in enumerate(predictions_train):
            if y1_train_orig[count] == predictions_train[count]:
                true_pred += 1
            else:
                false_pred += 1
        print('Total number of train-set predictions is: ' +
              str(len(y1_train_orig)))
        print('Number of correct train-set predictions is: ' + str(true_pred))
        print('Number of incorrect train-set predictions is: ' + str(false_pred))

        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns AUC
        auc_score = sklearn.metrics.roc_auc_score(
            y1_train_orig, predictions_train)
        print('train-set auc score is: ' + str(auc_score))
        print('train-set seed number is: ' + str(seed))

        ##########################################################
        # Apply on test data
        pred_test = model.predict({'forward': x1_test, 'reverse': x2_test})
        # See which label has the highest confidence value
        predictions_test = np.argmax(pred_test, axis=1)

        true_pred = 0
        false_pred = 0
        for count, value in enumerate(predictions_test):
            if y1_test_orig[count] == predictions_test[count]:
                true_pred += 1
            else:
                false_pred += 1
        print('Total number of test-set predictions is: ' + str(len(y1_test_orig)))
        print('Number of correct test-set predictions is: ' + str(true_pred))
        print('Number of incorrect test-set predictions is: ' + str(false_pred))

        auc_score = sklearn.metrics.roc_auc_score(
            y1_test_orig, predictions_test)
        print('test-set auc score is: ' + str(auc_score))
        print('test-set seed number is: ' + str(seed))
        sys.exit()
        #######*******************************

        """
        model.load_weights(weight_dir)
        weights = model.get_weights()

        # Apply on test data
        pred_test = model.predict({'forward': x1_test, 'reverse': x2_test})
        # Sum the absolute difference between y1_test and pred_test
        vals = np.sum(np.absolute(np.subtract(y1_test, pred_test)), axis=1)
        baseline = np.average(vals)
        """
        distances = []
        for i in range(self.filters):
            model.load_weights(weight_dir)
            weights = model.get_weights()

            zeros = np.zeros((12, 4))
            weights[0][:, :, i] = zeros
            model.set_weights(weights)

            ##########################################################
            # Apply on test data
            pred_test = model.predict({'forward': x1_test, 'reverse': x2_test})
            # See which label has the highest confidence value
            vals = np.sum(np.absolute(np.subtract(y1_test, pred_test)), axis=1)
            ave_distance = np.average(vals)
            distances.append(ave_distance)
            print(i)
        print(distances)
        np.savetxt('distances.txt', distances)

    def eval(self):
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()

        # if want dinucleotide sequences
        # dict = prep.dinucleotide_encode()

        # print maximum length without truncation
        np.set_printoptions(threshold=sys.maxsize)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        # seed = random.randint(1, 1000)

        x1_train, x1_test, y1_train, y1_test = train_test_split(
            fw_fasta, readout, test_size=0.1, random_state=seed)
        # split for reverse complemenet sequences
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            rc_fasta, readout, test_size=0.1, random_state=seed)
        #assert x1_test == x2_test
        #assert y1_test == y2_test

        model = self.create_model()

        # change from list to numpy array
        y1_train = np.asarray(y1_train)
        y1_test = np.asarray(y1_test)
        y2_train = np.asarray(y2_train)
        y2_test = np.asarray(y2_test)

        # Copy the original target values for later uses
        y1_train_orig = y1_train.copy()
        y1_test_orig = y1_test.copy()

        # if we want to merge two training dataset
        # comb = np.concatenate((y1_train, y2_train))

        ## Change it to categorical values
        y1_train = keras.utils.to_categorical(y1_train, 2)
        y1_test = keras.utils.to_categorical(y1_test, 2)

        # train the data
        model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train,
                  epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)
        ## Save the entire model as a SavedModel.
        ##model.save('my_model')
        # Save weights only: later used in self.filter_importance()
        #model.save_weights('./my_checkpoint')

        # save each convolution learned filters as txt file
        """
        motif_weight = model.get_weights()
        motif_weight = np.asarray(motif_weight[0])
        for i in range(int(self.filters)):
            x = motif_weight[:,:,i]
            berd = np.divide(np.exp(100*x), np.transpose(np.expand_dims(np.sum(np.exp(100*x), axis = 1), axis = 0), [1,0]))
            np.savetxt(os.path.join('./motif_files', 'filter_num_%d'%i+'.txt'), berd)
        """
        pred_train = model.predict({'forward': x1_train, 'reverse': x2_train})

        # See which label has the highest confidence value
        predictions_train = np.argmax(pred_train, axis=1)

        print(y1_train_orig[0:10])
        print(predictions_train[0:10])

        true_pred = 0
        false_pred = 0
        for count, value in enumerate(predictions_train):
            if y1_train_orig[count] == predictions_train[count]:
                true_pred += 1
            else:
                false_pred += 1
        print('Total number of train-set predictions is: ' +
              str(len(y1_train_orig)))
        print('Number of correct train-set predictions is: ' + str(true_pred))
        print('Number of incorrect train-set predictions is: ' + str(false_pred))

        # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
        # Returns AUC
        auc_score = sklearn.metrics.roc_auc_score(
            y1_train_orig, predictions_train)
        print('train-set auc score is: ' + str(auc_score))
        print('train-set seed number is: ' + str(seed))

        ##########################################################
        # Apply on test data
        pred_test = model.predict({'forward': x1_test, 'reverse': x2_test})
        # See which label has the highest confidence value
        predictions_test = np.argmax(pred_test, axis=1)

        true_pred = 0
        false_pred = 0

        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        for count, value in enumerate(predictions_test):
            if y1_test_orig[count] == predictions_test[count]:
                true_pred += 1
                if value == 1:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                false_pred += 1
                if value == 1:
                    false_pos += 1
                else:
                    false_neg += 1

        print('Total number of test-set predictions is: ' + str(len(y1_test_orig)))
        print('Number of correct test-set predictions is: ' + str(true_pred))
        print('Number of incorrect test-set predictions is: ' + str(false_pred))

        print('True Positives:', true_pos)
        print('True Negatives:', true_neg)
        print('False Positives:', false_pos)
        print('False Negatives:', false_neg)

        precision = true_pos/(true_pos+false_pos)
        recall = true_pos/(true_pos+false_neg)
        print('Precision:', precision)
        print('Recall:', recall)

        f1 = 2 / (1 / precision + 1 / recall)
        print('F1 Score:', f1)

        auc_score = sklearn.metrics.roc_auc_score(
            y1_test_orig, predictions_test)
        print('test-set auc score is: ' + str(auc_score))
        print('test-set seed number is: ' + str(seed))

    def hyperopt_tuner(self):
        def objective(param):
            prep = preprocess(self.fasta_file, self.readout_file)

            # if want mono-nucleotide sequences
            dict = prep.one_hot_encode()

            # if want dinucleotide sequences
            # dict = prep.dinucleotide_encode()

            # print maximum length without truncation
            np.set_printoptions(threshold=sys.maxsize)

            fw_fasta = dict["forward"]
            rc_fasta = dict["reverse"]
            readout = dict["readout"]

            # seed = random.randint(1, 1000)

            x1_train, x1_test, y1_train, y1_test = train_test_split(
                fw_fasta, readout, test_size=0.1, random_state=seed)
            # split for reverse complemenet sequences
            x2_train, x2_test, y2_train, y2_test = train_test_split(
                rc_fasta, readout, test_size=0.1, random_state=seed)
            #assert x1_test == x2_test
            #assert y1_test == y2_test

            model = self.create_model()

            # change from list to numpy array
            y1_train = np.asarray(y1_train)
            y1_test = np.asarray(y1_test)
            y2_train = np.asarray(y2_train)
            y2_test = np.asarray(y2_test)

            # if we want to merge two training dataset
            # comb = np.concatenate((y1_train, y2_train))

            # train the data
            model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train,
                      epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)

            history2 = model.evaluate(
                {'forward': x1_test, 'reverse': x2_test}, y1_test)

            return history2[0]

        parameter = dict(kernel_size=hp.choice('kernel_size', [12, 16]),
                         batch_size=hp.choice('batch_size', [512]),
                         epochs=hp.choice('epochs', [30, 40, 50]),
                         filters=hp.choice('filters', [16, 256]))

        # calling the hyperopt function and setting the maximum iteration to 100
        best_params = fmin(objective,
                           parameter,
                           algo=tpe.suggest,
                           max_evals=5)

        print(best_params)

    def cross_val(self):
        # so that we get different metrics used in this custom version
        # preprocess the data
        prep = preprocess(self.fasta_file, self.readout_file)

        # if want mono-nucleotide sequences
        dict = prep.one_hot_encode()
        # if want dinucleotide sequences
        #dict = prep.dinucleotide_encode()

        np.set_printoptions(threshold=sys.maxsize)

        # seed to reproduce results
        # seed = random.randint(1, 1000)

        fw_fasta = dict["forward"]
        rc_fasta = dict["reverse"]
        readout = dict["readout"]

        #if self.activation_type == 'linear':
        #    readout = np.log2(readout)
        #    readout = np.ndarray.tolist(readout)

        forward_shuffle, readout_shuffle = shuffle(
            fw_fasta, readout, random_state=seed)
        reverse_shuffle, readout_shuffle = shuffle(
            rc_fasta, readout, random_state=seed)
        readout_shuffle = np.array(readout_shuffle)

        # initialize metrics to save values
        metrics = []

        # save the information of 10 folds auc scores
        train_auc_scores = []
        test_auc_scores = []

        # Provides train/test indices to split data in train/test sets.
        kFold = StratifiedKFold(n_splits=10)
        ln = np.zeros(len(readout_shuffle))
        for train, test in kFold.split(ln, ln):
            model = None
            model, model2 = self.create_model()

            fwd_train = forward_shuffle[train]
            fwd_test = forward_shuffle[test]
            rc_train = reverse_shuffle[train]
            rc_test = reverse_shuffle[test]
            y_train = readout_shuffle[train]
            y_test = readout_shuffle[test]

            model = self.create_model()

            # Early stopping
            callback = EarlyStopping(monitor='loss', min_delta=0.001, patience=3,
                                     verbose=0, mode='auto', baseline=None, restore_best_weights=False)
            history = model.fit({'forward': fwd_train, 'reverse': rc_train}, y_train, epochs=self.epochs,
                                batch_size=self.batch_size, validation_split=0.0, callbacks=[callback])

            # Without early stopping
            model.fit({'forward': x1_train, 'reverse': x2_train}, y1_train,
                      epochs=self.epochs, batch_size=self.batch_size, validation_split=0.0)

            pred_train = model.predict(
                {'forward': x1_test, 'reverse': x2_test})
            vals = []

            for i in range(len(pred_train)):
                if pred_train[i] < 0.5:
                    val = 0
                    vals.append(val)
                if pred_train[i] >= 0.5:
                    val = 1
                    vals.append(val)

            print(y1_train[0:10])
            print(vals[0:10])

            true_pred = 0
            false_pred = 0
            for ind in range(len(pred_train)):
                if y1_train[ind] == vals[ind]:
                    true_pred += 1
                else:
                    false_pred += 1
            print('Total number of train-set predictions is: ' + str(len(y1_train)))
            print('Number of correct train-set predictions is: ' + str(true_pred))
            print('Number of incorrect train-set predictions is: ' + str(false_pred))

            auc_score = sklearn.metrics.roc_auc_score(y1_train, pred_train)
            print('train-set auc score is: ' + str(auc_score))
            print('train-set seed number is: ' + str(seed))
            train_auc_scores.append(auc_score)

            ##########################################################

            pred = model.predict({'forward': x1_test, 'reverse': x2_test})

            vals = []
            for i in range(len(pred)):
                if pred[i] < 0.5:
                    val = 0
                    vals.append(val)
                if pred[i] >= 0.5:
                    val = 1
                    vals.append(val)

            true_pred = 0
            false_pred = 0
            for ind in range(len(y1_test)):
                if y1_test[ind] == vals[ind]:
                    true_pred += 1
                else:
                    false_pred += 1
            print('Total number of test-set predictions is: ' + str(len(y1_test)))
            print('Number of correct test-set predictions is: ' + str(true_pred))
            print('Number of incorrect test-set predictions is: ' + str(false_pred))

            auc_score = sklearn.metrics.roc_auc_score(y1_test, pred)
            print('test-set auc score is: ' + str(auc_score))
            print('test-set seed number is: ' + str(seed))
            test_auc_scores.append(auc_score)

        print('seed number = %d' % seed)
        print(train_auc_scores)
        print('Mean train auc_scores of 10-fold cv is ' +
              str(np.mean(train_auc_scores)))
        print(test_auc_scores)
        print('Mean test auc_scores of 10-fold cv is ' +
              str(np.mean(test_auc_scores)))
