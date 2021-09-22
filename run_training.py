import os
import json
import tensorflow as tf

from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib as mpl

from dataloader import *
from data_preprocess import *
from training_model import *

import datetime
import argparse
import sys

if __name__ == '__main__':

    """
    Usage Notes

    --mode : specifies if you  want to train a new model or test existing one
    python3 run_training.py --mode train
    """
    mpl.use('tkagg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    args = parser.parse_args()

    mode = args.mode

    f = open('./configs.json', 'r')
    configs = json.load(f)

    CHAR_VECTOR = configs['CHAR_VECTOR']
    letters = [letter for letter in CHAR_VECTOR]

    mnist_image_height = configs['image_height']
    mnist_image_width = configs['image_width']
    max_seq_len = configs['max_seq_len']

    synth_img_height = 28
    synth_img_width = 28 * max_seq_len

    imgSize = (synth_img_height, synth_img_width, 1)

    num_classes = configs['num_classes']
    checkpoints_path = configs['checkpoints_path']

    #Load MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    #Checking the downloaded data
    print("Shape of original training dataset: {}".format(np.shape(X_train)))
    print("Shape of origninal test dataset: {}".format(np.shape(X_test)))

    if mode == 'train':
        # Generate Synthetic dataset
        X_synth_train,y_synth_train = build_synth_data(X_train,y_train,60000,max_seq_len,mnist_image_height,mnist_image_width,synth_img_height,synth_img_height)
        # X_synth_test,y_synth_test = build_synth_data(X_test,y_test,10000,max_seq_len,mnist_image_height,mnist_image_width,synth_img_height,synth_img_height)
        X_synth_val,y_synth_val = build_synth_data(X_test,y_test,20000,max_seq_len,mnist_image_height,mnist_image_width,synth_img_height,synth_img_height)

        #checking a sample
        plt.figure()
        plt.imshow(X_synth_train[30], cmap='gray')
        plt.title("This image represents the sequence: " + str(y_synth_train[30]))
        plt.axis('off')
        #print("This image represents the sequence: ", y_synth_test[30])
        #print("Shape of Training set: ", X_synth_train.shape)
        plt.waitforbuttonpress()

        train_images = prep_data_keras(X_synth_train,synth_img_height, synth_img_width)
        # test_images = prep_data_keras(X_synth_test,synth_img_height, synth_img_width)
        val_images = prep_data_keras(X_synth_val,synth_img_height, synth_img_width)

        # Set data configs
        downsample_factor = 2 * 1
        batch_size = configs['batch_size']
        val_batch_size = configs['val_batch_size']

        # Load dataset
        tiger_train = TextImageGenerator(synth_img_width, synth_img_height, batch_size, downsample_factor, train_images, y_synth_train)
        # tiger_test = TextImageGenerator(synth_img_width, synth_img_height, val_batch_size, downsample_factor, test_images, y_synth_test)
        tiger_val = TextImageGenerator(synth_img_width, synth_img_height, val_batch_size, downsample_factor, val_images, y_synth_val)


        # Add Tensorboard Callback
        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        #Add checkpoints
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,
                                                     save_weights_only=True,
                                                     verbose=1)

        epochs = configs['epochs']
        training_model  = model(imgSize, num_classes, max_seq_len, mode)
        training_model.summary()

        training_model.fit(tiger_train.next_batch(), steps_per_epoch=int(tiger_train.n / batch_size),
                       validation_data=tiger_val.next_batch(), validation_steps=int(tiger_val.n / val_batch_size), epochs=epochs,
                       callbacks=[cp_callback, tensorboard_callback])

    if mode == 'test':

        if not os.path.exists(checkpoints_path):
            print("Trained model not found. Please use --mode train before testing")
            sys.exit()

        X_synth_test,y_synth_test = build_synth_data(X_test,y_test,10000,max_seq_len,mnist_image_height,mnist_image_width,synth_img_height,synth_img_height)
        test_images = prep_data_keras(X_synth_test,synth_img_height, synth_img_width)
        tiger_test = TextImageGenerator(synth_img_width, synth_img_height, val_batch_size, downsample_factor, test_images, y_synth_test)

        pred_list = []
        gt_list = []

        test_model = model(imgSize, num_classes, max_seq_len, mode)
        test_model.load_weights(ckpt_path)

        total = 0
        count = 0
        for i in range(X_synth_test.shape[0]):

            img_data = X_synth_test[i].reshape(-1,synth_img_height, synth_img_width,1)
            net_out_value = test_model.predict(img_data)
            pred_texts = decode_label(net_out_value)

            ground_truth = y_synth_test[i]#.tolist()


            pred_texts = ''.join(pred_texts)
            ground_truth = ''.join(ground_truth)

            pred_list.append(pred_texts)
            gt_list.append(ground_truth)

            ed = editdistance.eval(pred_texts, ground_truth)

            total += ed
            count += 1

        avg_ed = total/count
        print("SAMPLE PREDICTIONS")
        for i in range(5):
            print("GROUND TRUTH: ", gt_list[i])
            print("PREDICTED TEXT: ", pred_list[i])

        print("AVERAGE EDIT DISTANCE ON TEST SET: ", avg_ed)
