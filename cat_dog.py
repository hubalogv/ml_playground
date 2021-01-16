# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16


def build_custom_model():
    # Initialising the CNN
    mod = Sequential()
    # Step 1 - Convolution
    mod.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    # Step 2 - Pooling
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Adding a second convolutional layer
    mod.add(layers.Conv2D(32, (3, 3), activation='relu'))
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # Step 3 - Flattening
    mod.add(layers.Flatten())
    # Step 4 - Full connection
    mod.add(layers.Dense(units=128, activation='relu'))
    mod.add(layers.Dense(units=1, activation='sigmoid'))
    # Compiling the CNN
    mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return mod

def build_resnet50():
    mod = Sequential()
    res = ResNet50(include_top=False, input_shape=(64, 64, 3), pooling='avg', weights='imagenet')
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


if __name__ == '__main__':

    model = build_resnet50()
    # model = build_custom_model()

    data_path = r'C:\_ws\datasets\cat_dog'
    batch_size = 32
    target_size = (256, 256)

    train_data_gen = ImageDataGenerator(validation_split=0.2,
                                        rescale=1./255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        brightness_range=(0.8, 1.2),
                                        rotation_range=40,
                                        horizontal_flip=True)


    training_set = train_data_gen.flow_from_directory(data_path + r'\training_set',
                                                      subset='training',
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      class_mode='binary')
    validation_set = train_data_gen.flow_from_directory(data_path + r'\training_set',
                                                        subset='validation',
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        class_mode='binary')

    es_callback = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=9,
        min_delta=0.001,
        restore_best_weights=True)

    model.fit(training_set,
              steps_per_epoch=int(training_set.samples / batch_size),
              epochs=100,
              validation_data=validation_set,
              validation_steps=int(validation_set.samples / batch_size),
              callbacks=[es_callback, callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=5)])
    # Part 3 - Making new predictions

    if 1:
        result = model.evaluate(validation_set)
        print(result)

    if 0:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(data_path + r'\test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')
        result = model.evaluate(test_set)
        print(result)
