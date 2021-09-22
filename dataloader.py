import random
import numpy as np
from skimage.transform import resize
from data_preprocess import text_to_labels

class TextImageGenerator:
    def __init__(self, img_w, img_h,
                 batch_size, downsample_factor, imgs, texts, max_seq_len=5):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.downsample_factor = downsample_factor
        self.n = imgs.shape[0]                                                                            # Number of images in train
        self.imgs = imgs
        self.texts = texts
        self.indexes = list(range(self.n))
        self.cur_index = 0



    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_h, self.img_w, 1])                                  # (bs, 28, 28 * max_seq_len, 1)
            Y_data = np.ones([self.batch_size, self.max_seq_len])                                           # (bs, max_seq_len)
            input_length = np.ones((self.batch_size, 1)) * (self.img_h // self.downsample_factor - 2)       # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))                                                   # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()

                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                label_length[i] = len(text)

            # dict
            inputs = {
                'the_input': X_data,                                                                        # (bs, 28, 28 * max_seq_len, 1)
                'the_labels': Y_data,                                                                       # (bs, max_seq_len)
                'input_length': input_length,                                                               # (bs, 1)
                'label_length': label_length                                                                # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}

            yield (inputs, outputs)
