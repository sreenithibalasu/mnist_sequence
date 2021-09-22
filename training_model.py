import numpy as np
import tensorflow as tf

from keras.models import Sequential, Model

from keras.layers import Flatten, Reshape, Dense, Dropout, Input, InputLayer, Bidirectional, BatchNormalization, Activation, Dropout, TimeDistributed, LSTM, Lambda, Layer
from keras.layers.convolutional import Conv2D, MaxPooling2D

from tensorflow.keras.optimizers import Adam

from keras import backend as K

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]

    #print(len(y_pred, len(labels)))
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def model(imgSize, num_classes, max_seq_len, mode):
    inputs = Input(name='the_input', shape=imgSize)


    # First Layer: Conv (5x5) + Pool (2x2) - Output size: 14 x 70 x 64
    conv_1 = Conv2D(64, kernel_size=(5,5),  strides=(1,1), padding='same', activation='relu', dtype='float32')(inputs)
    pool_1 = MaxPooling2D(pool_size=(2,2), padding='same')(conv_1)

    # Second Layer: Conv (5x5) + Pool (1x2) - Output size: 14 x 35 x 128
    conv_2 = Conv2D(128, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(1,2), padding='same')(conv_2)

    # Third Layer: Conv (3x3) + Pool (1x2) + Simple Batch Norm - Output Size: 14 x 17 x 128
    conv_3 = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(1,2), padding='same')(conv_3)
    batch_3 = BatchNormalization()(pool_3)
    dropout_1 = Dropout(0.5)(batch_3)

    # Fourth Layer:Conv (3x3) + Pool (1x2) + Simple Batch Norm - Output Size: 14 x 8 x 256
    conv_4 = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(dropout_1)
    pool_4 = MaxPooling2D(pool_size=(1,2), padding='same')(conv_4)
    batch_4 = BatchNormalization()(pool_4)

    # Fifth Layer:Conv (3x3) + Pool (1x2) + Simple Batch Norm - Output Size: 14 x 4 x 256
    conv_5 = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(batch_4)
    pool_5 = MaxPooling2D(pool_size=(1,2), padding='same')(conv_5)
    batch_5 = BatchNormalization()(pool_5)
    dropout_2 = Dropout(0.5)(batch_5)


    # Sixth Layer:Conv (3x3) + Pool (1x2) + Simple Batch Norm - Output Size: 14 x 2 x 512
    conv_6 = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(dropout_2)
    pool_6 = MaxPooling2D(pool_size=(1,2), padding='same')(conv_6)
    batch_6 = BatchNormalization()(pool_6)
    dropout_3 = Dropout(0.5)(batch_6)

    # Seventh Layer:Conv (3x3) + Pool (1x2) + Simple Batch Norm - Output Size: 14 x 1 x 512
    conv_7 = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(dropout_3)
    pool_7 = MaxPooling2D(pool_size=(1,2), padding='same')(conv_7)

    # Collapse layer to remove dimension 14 x 2 x 512 ---> 14 x 1024 on axis=2
    reshape_8 = Reshape(target_shape=(14 ,1024))(pool_7)
    dense_8 = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(reshape_8)


    # 2 layers of LSTM cells used to build RNN
    numHidden = 512

    lstm_1 = Bidirectional(LSTM(units=numHidden, return_sequences=True))(dense_8)
    lstm_2 = Bidirectional(LSTM(units=numHidden, return_sequences=True))(lstm_1)
    batch_8 = BatchNormalization()(lstm_2)

    #Map to output
    #reshape_9 = Reshape((-1, 14, 1024))(lstm_1)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(batch_8)

    labels = Input(name='the_labels', shape=[max_seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

    #model to be used at training time
    if mode == 'train':
        training_model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

        ada = Adam(lr=0.0001)
        training_model.compile(optimizer=ada, loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])

        return training_model
    if mode == 'test':
        test_model = Model(inputs=[inputs], outputs=outputs)
        return test_model
