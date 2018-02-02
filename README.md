# Efficient Video Classification
This project provides a video classification module that has a classification accuracy of 60%-70% and can perform the video
classification task in less than 8 seconds for videos of any duration. This module can benefit mobile device computing as well as processing billions of videos on the fly.

# Features
- Uniform Video Frame Sampling
- Convolutional Neural Network Feature Extraction for Each Frame
- Video-Level Information Aggregation Using LSTM

# Files & Instructions
- extract_frames_uniform.py (can be found in the data folder): this is a utility tool that can be used to uniformally sample frames from each video. To use this uitlity tool, type in the command line:

"python extract_frames.uniform.py"

Once the frames are extracted, then the next step is to extract the CNN features for the sampled frames of a video. 

- extractor.py: Extractor object that extracts CNN feature using Inception V3 model or MobileNet model.
- extract_seq_features.py: this utility tool calls the extract methods in the Extractor object to extract CNN features for the uniformally sampled frames of each video. To use this uitlity tool, type in the command line:

"python extract_seq_features.py"

Once the CNN features are extracted for the frames of each video, then one needs to feed the features and the related class labels into the model for training and testing.

- data.py: creates a generator to generate batches for training and testing
- train_CNN_LSTM.py: this module calls the generator provided by data.py to get batches for training. In particular, it trains an LSTM model given the input CNN features and the class labels. To train the model, type in the command line:

"python train_CNN_LSTM.py"

- train_CNN.py: this utility tool allows you to fine tune the Inception or MobileNet model using your own dataset. To use this utility tool, type in the command line:

"python train_CNN.py"
