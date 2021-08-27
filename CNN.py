import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATA_PATH = "/content/data.json"


def load_data(data_path):
   

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
  
    fig, axs = plt.subplots(2)

    # graficul acuratetii

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Acuratetea")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Evaluarea Acuratetii")

    #  graficul erorii
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Eroarea")
    axs[1].set_xlabel("Epoca")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Evaluarea Erorii")

    plt.show()


def prepare_datasets(test_size, validation_size):


    # incarcam datele

    X, y = load_data(DATA_PATH)

    # impartim datele in date de antrenament, testare si validare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # Adaugam o noua axa, echivalenta cu numarul de canale pe care il au pozele
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    

    # arhitectura retelei
    model = keras.Sequential()

    # primul strat conv
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # al2lea strat conv
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # al3lea strat conv
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # aplatizare + strat conectat complet
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    # stratul de iesire
    model.add(keras.layers.Dense(4, activation='softmax'))

    return model


def predict(model, X, y):
   

    # Adaugam o noua axa 
    X = X[np.newaxis, ...] 

    # realizam predictia cu functia predict din keras
    prediction = model.predict(X)

    # luam indexul cu valoarea cea mai mare
    predicted_index = np.argmax(prediction, axis=1)
    #ordinea este asa datorita modului in care au fost citite folderele din drive
    print("Target: {}, Predicted label: {}".format(y, predicted_index))
    if predicted_index == 0:
        print("Instrumentul este Vioara")
    elif predicted_index == 1:
            print("Instrumentul este Flaut")
    elif predicted_index == 2:
            print("Instrumentul este Pian")
    else:
            print("Instrumentul este Trompeta")
        

if __name__ == "__main__":

    # impartim datele in date de antrenament, testare si validare
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)

    # cream reteaua
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)

    # compilam modelul
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # antrenam modelul
    
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=50)
    t2=ti.time()
    print(t2-t1)

    # expunem graficul erorilor si al acuratetii
    plot_history(history)

    # evaluam modelul pe lotul de testt
    t1=ti.time()
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    t2=ti.time()
    print('\nAcuratetea este de:', test_acc*100,'%')
    print ('\nTimp predictie pe tot setul de test: ',t2-t1)
