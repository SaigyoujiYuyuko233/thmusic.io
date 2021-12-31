from util import sample
import tensorflow as tf
import tensorflow.keras as keras
import glob
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models
from random import randrange
import datetime

if __name__ == '__main__':
    CLIP_SIZE = 120000

    train_spect = []
    train_label = []
    for file in glob.glob("./dataset-compressed/train/*/*.mp3"):
        print("Loading file [" + file + "]")
        if "nonsakurasakura" in file:
            spect, label = sample.gen_train_sample(file, label=1, clip_size=CLIP_SIZE)
        else:
            spect, label = sample.gen_train_sample(file, label=0, clip_size=CLIP_SIZE)
        train_spect += spect
        train_label += label
    train_spect = tf.convert_to_tensor(np.array(train_spect))
    train_spect = tf.reshape(train_spect, [train_spect.shape[0], train_spect.shape[1], train_spect.shape[2], 1])
    train_label = tf.convert_to_tensor(np.array(train_label))
    print(train_spect.shape)
    print(train_label.shape)

    test_spect = []
    test_label = []
    for file in glob.glob("./dataset-compressed/test/*/*.mp3"):
        print("Loading file [" + file + "]")
        if "nonsakurasakura" in file:
            spect, label = sample.gen_train_sample(file, label=1, clip_size=CLIP_SIZE)
        else:
            spect, label = sample.gen_train_sample(file, label=0, clip_size=CLIP_SIZE)
        test_spect += spect
        test_label += label
    test_spect = tf.convert_to_tensor(np.array(test_spect))
    test_spect = tf.reshape(test_spect, [test_spect.shape[0], test_spect.shape[1], test_spect.shape[2], 1])
    test_label = tf.convert_to_tensor(np.array(test_label))
    print(test_spect.shape)
    print(test_label.shape)

    for i in range(2):
        index = randrange(0, len(test_spect))
        plt.text(0, 0, "Test | Label: " + str(test_label[index].numpy()))
        plt.imshow(test_spect[index].numpy())
        plt.show()
        index = randrange(0, len(train_spect))
        plt.text(0, 0, "Train | Label: " + str(train_label[index].numpy()))
        plt.imshow(train_spect[index].numpy())
        plt.show()

    # 创建模型
    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=[train_spect.shape[1], train_spect.shape[2], 1]),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 0: sakurasakura
    # 1: 蓬莱传说
    EPOCHS = 200
    steps_per_epoch = 200
    batch_size = int(train_spect.shape[0] / steps_per_epoch)
    history = model.fit(train_spect, train_label, validation_data=(test_spect, test_label), batch_size=batch_size,
                        steps_per_epoch=steps_per_epoch, epochs=EPOCHS, callbacks=[tensorboard_callback])
    # history = model.fit(train_spect, train_label, epochs=12)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.2, 1])
    plt.legend(loc='lower right')
    plt.show()

    model.save('models/model_' + str(
        history.history['accuracy'][len(history.history['accuracy']) - 1]) + "_" + datetime.datetime.now().strftime(
        "%Y_%m_%d-%H_%M_%S") + ".h5")
