import csv
import numpy as np
import random
import os.path
import sys
from keras.utils import to_categorical
import threading

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

classes = ['2', '29', '26', '43', '30', '36', '37', '44', '23', '15', '10', '39', '22', 
           '24', '28', '17', '27', '1', '42', '41', '34', '35', '20', '25', '33', '19']

class_dict = {}
for key, val in enumerate(classes):
    class_dict[val] = key

print(class_dict)
    
seq_length= 5

def get_data(filename):
    with open(filename, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
         
        return data

@threadsafe_generator
def frame_generator_by_group(batch_size, group, video_list, feature_type='original'):
    data = get_data(video_list)
    while 1:
        X, y = [], []
        for _ in range(batch_size):
            sample = random.choice(data)

            path = None
            if feature_type == 'original':
                path = os.path.join('data','sequences', group, sample[1]+ '_' + str(seq_length) + '-features.npy')
            elif feature_type == 'multires':
                path = os.path.join('data','sequences', group, 'multires', sample[1]+ '_multires_' + str(seq_length) + '-features.npy')
            else:
                print("the feature type is not specified.")
                sys.exit()

            sequence = None
            if os.path.isfile(path):
                sequence = np.load(path)
            else:
                print(path, " does not exist")
                sys.exit()
            X.append(sequence)
            y.append(to_categorical(class_dict[sample[0]], len(classes)))
        X = np.array(X)
        y = np.array(y)

        yield X, y
