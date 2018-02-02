from keras.applications import*
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

checkpointer = ModelCheckpoint(
    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

early_stopper = EarlyStopping(patience=20)

tensorboard = TensorBoard(log_dir='./data/logs/')

classes = ['2', '29', '26', '43', '30', '36', '37', '44', '23', '15', '10', '39', '22', 
           '24', '28', '17', '27', '1', '42', '41', '34', '35', '20', '25', '33', '19']

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        './data/uniform_data/train/',
        target_size=(299, 299),
        batch_size=64,
        classes=classes,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        './data/uniform_data/test/',
        target_size=(299, 299),
        batch_size=64,
        classes=classes,
        class_mode='categorical')

    return train_generator, validation_generator

def get_model():
    weights='imagenet'
    base_model = InceptionV3(weights=weights, include_top=False)
    #base_model = InceptionResNetV2(weights=weights, include_top=False)
    #base_model = Xception(weights=weights, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(classes), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def fine_tune_model(model):
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
            
    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def train_model(model, nb_epoch,generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=300,
        validation_data=validation_generator,
        validation_steps=50,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

if __name__ == '__main__':
    model = get_model()
    generators = get_generators()

    print("Training Top layers.")
    model = train_model(model, 2, generators)

    model = fine_tune_model(model)
    model = train_model(model, 10, generators,[checkpointer, early_stopper, tensorboard])
