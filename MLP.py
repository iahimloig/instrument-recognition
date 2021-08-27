import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt


DATA_PATH = "/content/data.json"



def load_data(data_path):
   

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convertim listele in vectori numpy
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Datele au fost incarcate!")

    return X, y


def plot_history(history):
    

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Acuratetea")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Evaluarea Acuratetii")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Eroarea")
    axs[1].set_xlabel("Epoca")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Evaluarea Erorii")

    plt.show()


if __name__ == "__main__":

    # incarcam datele
    X, y = load_data(DATA_PATH)

    # impartim datele in date de antrenament, testare si validare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # aritectura retelei
    model = keras.Sequential([

        # Stratul de intrare
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # Primul strat intermediar
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),

        # al2lea strat intermediar
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),

        # al3lea strat intermediar
        keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.4),

        # stratul de iesire
        keras.layers.Dense(4, activation='softmax')
    ])

    # compilam modelul
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # antrenam modelul
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)

    # expunem graficul erorilor si al acuratetii in functie de epoci
    plot_history(history)

    # evaluam modelul pe lotul de test
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nAcuratetea este de:', test_acc*100,'%.')
