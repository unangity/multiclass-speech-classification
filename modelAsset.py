import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras

from keras.models import Sequential
import keras.layers as layers



def plot_loss(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = list(range(1, len(val_loss) + 1))
    plt.plot(x, val_loss, color = 'red', label = 'Validation loss')
    plt.plot(x, train_loss, label = 'Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

def model(opt):
    nns = Sequential()

    nns.add(layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu', 
                        input_shape = (99, 13, 1)))
    nns.add(layers.MaxPooling2D((2, 2)))
    nns.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    nns.add(layers.MaxPooling2D((2, 2)))
    nns.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
    nns.add(layers.MaxPooling2D((2, 2)))
    nns.add(layers.Dense(64, activation = 'relu'))
    nns.add(layers.Conv2D(64, (3, 3), padding = 'same', activation = 'relu')) 
    nns.add(layers.Flatten())
    nns.add(layers.Dense(64, activation = 'relu'))
    nns.add(layers.Dropout(0.5))
    nns.add(layers.BatchNormalization())
    nns.add(layers.Dense(35, activation = 'softmax'))

    nns.compile(optimizer = opt, loss = 'categorical_crossentropy', 
                    metrics = ['accuracy'])

    return nns
