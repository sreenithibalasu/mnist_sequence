# MNIST Sequence Recognition

This repository contains code for:
- Generating a synthetic MNIST dataset
- Building a CNN-LSTM-CTC network to detect the sequence of digits in a given image

## Sample Sequence Image
![](https://github.com/sreenithibalasu/mnist_sequence/blob/master/images/Figure_2.png)

- A maximum sequence length of 5 was chosen to train my model
- The sequence length of synthetic images vary from 1-5. 
- If a sequence has less than 5 numbers, they are filled with 'x' (blank space character)

## Dataset
- The synthetic dataset consists of 60,000 images with image dimentsions 28 x 140
- The synthetic test dataset consists of 10,000 images of the same dimensions

## Training 
- The model was trained for 15 epochs with Adam optimizer
- After 15 epochs, I achieved a validation loss of 0.915 and validation accuracy of 78.8%
- This can be improved by generating more training data or by training for more epochs.

## Testing
- For training model performance on test set, I used the edit distance score
- Average Edit distance on test set = 2.87

## Configurations
- For training and generating the dataset on your machine, change the following in `configs.json`:
  - `max_seq_len`: maximum number of images in your synthetic dataset
  - `batch_size`: number of training examples used in one epoch
  - `val_batch_size`: number of validation examples used during one training epoch
  - `logs_path`: path to store logs for Tensorboard
  - `checkpoints_path`: path to store model weights after training. Done to load an existing model for testing
  - `epochs`: number of training iterations

Before training, create a virtual environment and run `pip install -r requirements.txt` after cloning the repository.
