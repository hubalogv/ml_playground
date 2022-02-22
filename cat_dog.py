# Importing the Keras libraries and packages
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16


random_seed = 2

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

def build_custom_model(input_shape):
    # Initialising the CNN
    mod = Sequential()
    # Step 1 - Convolution
    mod.add(layers.Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
    # Step 2 - Pooling
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    mod.add(layers.Conv2D(64, (3, 3), activation='relu'))
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    mod.add(layers.Flatten())
    # Step 4 - Full connection
    mod.add(layers.Dense(units=128, activation='relu'))
    mod.add(layers.Dense(units=1, activation='sigmoid'))
    # Compiling the CNN
    mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return mod

def build_resnet50(input_shape):
    mod = Sequential()
    res = ResNet50(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    # res.trainable = False
    mod.add(res)
    # mod.add(layers.Flatten())
    # mod.add(layers.Dense(units=64, activation='relu'))
    # mod.add(layers.Dense(units=64, activation='relu'))
    # mod.add(layers.Dropout(0.2))
    # mod.add(layers.GlobalAveragePooling2D())
    # mod.add(layers.Dropout(0.2))
    mod.add(layers.Dense(units=128, activation='relu'))
    mod.add(layers.Dropout(0.2))
    mod.add(layers.Dense(units=128, activation='relu'))
    mod.add(layers.Dropout(
        0.2))
    mod.add(layers.Dense(units=1, activation='sigmoid'))

    # mod.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    mod.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return mod


class pipeline(object):

    def __init__(self):
        self.target_size = (64, 64, 3)
        self.batch_size = 32
        self.model = None

    def build_model(self):
        # self.model = build_resnet50(self.target_size)
        self.model = build_custom_model(self.target_size)

    def train(self):

        self.data_path = r'C:\_ws\datasets\cat_dog'

        train_data_gen = ImageDataGenerator(validation_split=0.2,
                                            rescale=1./255,
                                            shear_range=0.2,
                                            zoom_range=0.2,
                                            brightness_range=(0.8, 1.2),
                                            rotation_range=40,
                                            horizontal_flip=True)


        training_set = train_data_gen.flow_from_directory(self.data_path + r'\training_set',
                                                          subset='training',
                                                          target_size=self.target_size[0:2],
                                                          batch_size=self.batch_size,
                                                          class_mode='binary',
                                                          # seed=random_seed,
                                                          # shuffle=False
                                                          )
        self.validation_set = train_data_gen.flow_from_directory(self.data_path + r'\training_set',
                                                            subset='validation',
                                                            target_size=self.target_size[0:2],
                                                            batch_size=self.batch_size,
                                                            class_mode='binary',
                                                            # seed=random_seed,
                                                            # shuffle=False
                                                            )

        es_callback = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=9,
            min_delta=0.001,
            restore_best_weights=True)

        self.model.fit(training_set,
                  steps_per_epoch=int(training_set.samples / self.batch_size),
                  epochs=100,
                  validation_data=self.validation_set,
                  validation_steps=int(self.validation_set.samples / self.batch_size),
                  callbacks=[es_callback, callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5)])
        # Part 3 - Making new predictions

        self.model.save(r"output/catdog/models/resnet.h5")

    def evaluate(self):
            result = self.model.evaluate(self.validation_set)
            print(result)

    def test(self):
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(self.data_path + r'\test_set',
                                                    target_size=self.target_size[0:2],
                                                    batch_size=self.batch_size,
                                                    class_mode='binary')
        result = self.model.evaluate(test_set)
        print(result)

if __name__ == '__main__':
    p = pipeline()
    p.build_model()
    p.train()
    p.evaluate()
    p.test()