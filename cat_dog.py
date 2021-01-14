# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_custom_model():
    # Initialising the CNN
    mod = Sequential()
    # Step 1 - Convolution
    mod.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    # Step 2 - Pooling
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    mod.add(Conv2D(32, (3, 3), activation='relu'))
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    mod.add(Flatten())
    # Step 4 - Full connection
    mod.add(Dense(units=128, activation='relu'))
    mod.add(Dense(units=1, activation='sigmoid'))
    # Compiling the CNN
    mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Part 2 - Fitting the CNN to the images
    return mod


if __name__ == '__main__':

    model = build_custom_model()

    data_path = r'C:\_ws\datasets\cat_dog'

    train_data_gen = ImageDataGenerator(validation_split=0.2,
                                        rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)

    batch_size = 32

    training_set = train_data_gen.flow_from_directory(data_path + r'\training_set',
                                                      subset='training',
                                                      target_size=(64, 64),
                                                      batch_size=batch_size,
                                                      class_mode='binary')
    validation_set = train_data_gen.flow_from_directory(data_path + r'\training_set',
                                                        subset='validation',
                                                        target_size=(64, 64),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    model.fit(training_set,
              steps_per_epoch=int(training_set.samples / batch_size),
              epochs=25,
              validation_data=validation_set,
              validation_steps=int(validation_set.samples / batch_size))
    # Part 3 - Making new predictions
    if 1:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(data_path + r'\test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
        result = model.evaluate(test_set)
        print(result)
