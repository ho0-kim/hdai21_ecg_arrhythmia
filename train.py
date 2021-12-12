import argparse

import numpy as np
from sklearn.utils import shuffle

np.random.seed(777)
import tensorflow as tf
tf.random.set_seed(777)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Add, Flatten, Activation, Dropout
from tensorflow.keras.optimizers import Adam

from read_data import read_data
from preprocess import minmixscale, elevation, augment, add_gaussian_noise

parser = argparse.ArgumentParser(prog='train.py', 
                                description='Train ECG data')

parser.add_argument('-d', '--data', type=str, required=True,
                    help='File path of training data')
parser.add_argument('-s', '--save', default='model.h5', type=str,
                    help='File name for saving trained model')
parser.add_argument('-b', '--batch', default=500, type=int,
                    help='Batch size (default=500)')
parser.add_argument('-e', '--epoch', default=50, type=int,
                    help='Number of epochs (default=50)')                    
parser.add_argument('-l', '--lead', default=2, type=int,
                    help='Number of leads to be trained (default=2)')
parser.add_argument('-v', '--elevation', default=False, action='store_true',
                    help='Option for adjusting elevation')
parser.add_argument('-a', '--augmentation', default=False, action='store_true',
                    help='Option for data augmentation (stretching & amplifying)')
parser.add_argument('-n', '--noise', default=False, action='store_true',
                    help='Option for adding noise on data')

args = parser.parse_args()

def create_model(X_train):
    n_obs, feature, depth = X_train.shape

    inp = Input(shape=(feature, depth))
    C = Conv1D(filters=32, kernel_size=32, strides=1)(inp)

    C11 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(C)
    A11 = Activation("relu")(C11)
    C12 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(A11)
    S11 = Add()([C12, C])
    A12 = Activation("relu")(S11)
    M11 = MaxPooling1D(pool_size=5, strides=2)(A12)


    C21 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(M11)
    A21 = Activation("relu")(C21)
    C22 = Conv1D(filters=32, kernel_size=32, strides=1, padding='same')(A21)
    S21 = Add()([C22, M11])
    A22 = Activation("relu")(S21)
    M21 = MaxPooling1D(pool_size=5, strides=2)(A22)

    C31 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(M21)
    A31 = Activation("relu")(C31)
    C32 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(A31)
    S31 = Add()([C32, M21])
    A32 = Activation("relu")(S31)
    M31 = MaxPooling1D(pool_size=5, strides=2)(A32)

    C41 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(M31)
    A41 = Activation("relu")(C41)
    C42 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(A41)
    S41 = Add()([C42, M31])
    A42 = Activation("relu")(S41)
    M41 = MaxPooling1D(pool_size=5, strides=2)(A42)

    C51 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(M41)
    A51 = Activation("relu")(C51)
    C52 = Conv1D(filters=32, kernel_size=16, strides=1, padding='same')(A51)
    S51 = Add()([C52, M41])
    A52 = Activation("relu")(S51)
    M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

    F1 = Flatten()(M51)

    DR1 = Dropout(0.5)(F1)

    D1 = Dense(32)(DR1)
    A6 = Activation("relu")(D1)
    D2 = Dense(16)(A6)
    D3 = Dense(1)(D2)
    A7 = Activation('sigmoid')(D3)

    return Model(inputs=inp, outputs=A7)


if __name__ == '__main__':
    print("reading files and extracting data...")
    X_train, y_train = read_data(args.data, n_leads=args.lead)
    
    print(X_train.shape, y_train.shape)

    print("applying Min-Max scaier...")
    X_train = minmixscale(X_train)

    if args.elevation:
        print("applying elevation adjustment...")
        X_train = elevation(X_train, n_leads=args.lead)

    if args.augmentation:
        print("applying data augmentation...")
        aug_idx = np.random.choice(len(X_train), int(len(X_train)/3))
        X_aug = np.zeros((len(aug_idx), X_train.shape[1], X_train.shape[2]))
        y_aug = np.zeros((len(aug_idx),))
        for i, idx in enumerate(aug_idx):
            X_aug[i] = augment(X_train[idx], n_leads=args.lead)
            y_aug[i] = y_train[idx]
        X_train = np.vstack([X_train, X_aug])
        y_train = np.hstack([y_train, y_aug])

    if args.noise:
        print("applying gaussian noises...")
        X_train = add_gaussian_noise(X_train, n_leads=args.lead)

    X_train, y_train = shuffle(X_train, y_train, random_state=777)

    model = create_model(X_train)
    adam = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, 
                epochs=args.epoch, 
                batch_size=args.batch, 
                )

    model.save(args.save)