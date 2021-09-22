import random
import numpy as np
from skimage.transform import resize
import json

f = open('./configs.json', 'r')
configs = json.load(f)

CHAR_VECTOR = configs['CHAR_VECTOR']
letters = [letter for letter in CHAR_VECTOR]

def decode_label(out):
    # out : (1, 14, 11)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr
#decoded = decode_label(net_out_val)

def build_synth_data(data,labels,dataset_size,max_seq_len,mnist_image_height,mnist_image_width,
                        synth_img_height,synth_img_width):

    #Define synthetic image dimensions
    synth_img_height = 28
    synth_img_width = 28 * max_seq_len

    #Define synthetic data
    synth_data = np.ndarray(shape=(dataset_size,synth_img_height,synth_img_width),
                           dtype=np.float32)

    #Define synthetic labels
    synth_labels = []

    #For a loop till the size of the synthetic dataset
    for i in range(0,dataset_size):

        #Pick a random number of digits to be in the dataset
        num_digits = random.randint(1,max_seq_len)

        #Randomly sampling indices to extract digits + labels afterwards
        s_indices = [random.randint(0,len(data)-1) for p in range(0,num_digits)]

        #stitch images together
        new_image = np.hstack([data[index] for index in s_indices])
        #stitch the labels together

        new_label = ''
        for index in s_indices:
          new_label += str(labels[index])




        #Loop till number of digits - 5, to concatenate blanks images, and blank labels together
        for j in range(0,max_seq_len-num_digits):
            new_image = np.hstack([new_image,np.zeros(shape=(mnist_image_height,
                                                                   mnist_image_width))])
            new_label += 'x'

        #Resize image
        new_image = resize(new_image,(synth_img_height, synth_img_width))

        #Assign the image to synth_data
        synth_data[i,:,:] = new_image

        #Assign the label to synth_data
        synth_labels.append(np.array(new_label))

    synth_labels = np.array(synth_labels)
    #Return the synthetic dataset
    return synth_data,synth_labels

def prep_data_keras(img_data,synth_img_height, synth_img_width):

    #Reshaping data for keras, with tensorflow as backend
    img_data = img_data.reshape(-1,synth_img_height, synth_img_width,1)

    #Converting everything to floats
    img_data = img_data.astype('float32')

    #Normalizing values between 0 and 1
    img_data /= 255.0

    return img_data

def labels_to_text(labels):     # letters index -> text (string)
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):      # text to letters
    return list(map(lambda x: letters.index(x), text))
