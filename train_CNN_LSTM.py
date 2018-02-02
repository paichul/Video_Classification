from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam

import sys
import time
from data import frame_generator_by_group
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model = 'lstm'
seq_length = 5
features_length = 2048
timestamp = time.time()
csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
                                    str(timestamp) + '.log'))

checkpointer = ModelCheckpoint(
    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

early_stopper = EarlyStopping(patience=100)

tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))


classes = ['2', '29', '26', '43', '30', '36', '37', '44', '23', '15', '10', '39', '22', 
           '24', '28', '17', '27', '1', '42', '41', '34', '35', '20', '25', '33', '19']
           
def get_model():
    model = Sequential()
    model.add(LSTM(2048, return_sequences=False,
                   input_shape=(seq_length, features_length),
                   dropout=0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    lstm = get_model()
    batch_size = 32
    train_generator = frame_generator_by_group(batch_size, "train", "train_video_list.csv")
    val_generator = frame_generator_by_group(batch_size, "test", "test_video_list.csv")

    lstm = get_model()
    lstm.model.fit_generator(
        generator=train_generator,
        steps_per_epoch=1000,
        epochs=100,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=40,
        workers=4)
    
