import numpy as np
import os.path
from extractor import Extractor
from tqdm import tqdm
import pandas as pd
import sys

seq_length = 5
model = Extractor()

def get_samples(filename):
    df = pd.read_csv(filename)
    X = df['video_name'].values
    y = df['category_id'].values
    return X, y

def extract_seq_features_by_group(video_list_file, input_dir, group_name, feature_type='original'):
    train_X, train_y = get_samples(video_list_file)

    for video_name, category_id in zip(train_X, train_y):
        if category_id in [38]:
            continue
        sequence = []
        class_path = input_dir + str(category_id) + "/" + video_name
        if feature_type == 'original':
            save_path = os.path.join('data', 'sequences', group_name , video_name + '_' + str(seq_length) + \
                                     '-features')
        elif feature_type == 'multires':
            save_path = os.path.join('data', 'sequences', group_name, 'multires', video_name + '_multires_' + str(seq_length) + \
                                     '-features')
        else:
            print("the feature type is not specified")
            sys.exit()
            
        if os.path.isfile(save_path + '.npy'):
            continue

        for i in range(seq_length):
            image_path = class_path + "_" + str(i) + ".png"
            features = None
            if feature_type == 'multires':
                features = model.extract_multires(image_path)
            elif feature_type == 'original':                
                features = model.extract(image_path)
            else:
                print("the feature type is not specified.")
                sys.exit()
            sequence.append(features)
        
        np.save(save_path, sequence)

extract_seq_features_by_group("./data/train_video_list.csv", "./data/uniform_data/train/", "train")
extract_seq_features_by_group("./data/test_video_list.csv", "./data/uniform_data/test/", "test")

#extract_seq_features_by_group("./data/train_video_list.csv", "./data/uniform_data/train/", "train", "multires")
#extract_seq_features_by_group("./data/test_video_list.csv", "./data/uniform_data/test/", "test", "multires")
