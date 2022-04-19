import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx

# Decide if to load an existing model or to train a new one
train_new_model = True
NUM_RUNS = 20
if train_new_model:
    # Loading the MNIST data set with samples and splitting it

    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()      #Here x is the png and y is the clasification

    # Normalizing the data (making length = 1)
    # Aqui el axis es el valor maximo al que estamos normalizando todos los valores. Los valores de los pixeles van de 0-225 pero para q sea mas facil procesar la información vamos a poner todos los valores en  un rango del 0-1
    X_train = tf.keras.utils.normalize(X_train, axis=1)   
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax')) #Clasificación

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Training the model
    history = model.fit(X_train, y_train, epochs=10)
    history.history.keys()
    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Ploting the accuracy
    plt.plot(history.history['accuracy'])
    plt.title('model acurracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc = 'upper left')
    plt.savefig('Train.png')
    plt.show()

    # Saving the model
    model.save('handwritten_digits.model')

else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.model')

# model.summary() 

# Load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1

        def get_shape(model):
            weights_layer = model.get_weights()
            shapes = []
            for weights in weights_layer:
                shapes.append(weights.shape)
            return shapes


        def set_shape(weights, shapes):
            new_weights = []
            index = 0
            for shape in shapes:
                if (len(shape) > 1):
                    n_nodes = np.prod(shape) + index
                else:
                    n_nodes = shape[0] + index
                tmp = np.array(weights[index:n_nodes]).reshape(shape)
                new_weights.append(tmp)
                index = n_nodes
            return new_weights


        def evaluate_nn(W, shape, X_train=X_train, y_train=y_train):
            results = []
            for weights in W:
                model.set_weights(set_shape(weights, shape))
                score = model.evaluate(X_train, y_train, verbose=0)
                results.append(1 - score[1])
            return results


        shape = get_shape(model)
        x_max = 1.0 * np.ones(83)
        x_min = -1.0 * x_max
        bounds = (x_min, x_max)
        options = {'c1': 0.4, 'c2': 0.8, 'w': 0.4}
        optimizer = GlobalBestPSO(n_particles=25, dimensions=83,
                                  options=options, bounds=bounds)
        cost, pos = optimizer.optimize(evaluate_nn, 15, X_train=X_train, Y_train=y_train, shape=shape)
        model.set_weights(set_shape(pos, shape))
        score = model.evaluate(X_test, y_train)
        start_time = time.clock()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print("--- %s seconds ---" % (time.clock() - start_time))