import sys
import subprocess
# implement pip as a subprocess:
dependencies = ['numpy', 'matplotlib', 'tensorflow', 'keras', 'neptune', 'medmnist']
for dependency in dependencies:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', dependency])

# Dependencies
import numpy as np
from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import losses

from keras.layers import Input, Conv2D, MaxPooling2D , UpSampling2D, add, Input
from keras.models import Model
from keras import regularizers

from skimage.io import imshow
import neptune
run = neptune.init_run(project='momkeybomkey/Federated',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNDdlN2ZhNy00ZmJmLTQ4YjMtYTk0YS1lNmViZmZjZWRhNzUifQ==')

import random

# Data loader functions
def _collate_fn(data):
        xs = []

        for x, _ in data:
            x = np.array(x).astype(np.float32) / 255.
            xs.append(x)

        return np.array(xs)

def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    result = []

    while len(result) < total_size:
        result.append(index[i])
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)

    return result

def get_loader(dataset, random):
    total_size = len(dataset)
    print('Size', total_size)
    if random:
        index_generator = shuffle_iterator(range(total_size))
    else:
        index_generator = list(range(total_size))
    
    while True:
        data = []

        for _ in range(len(index_generator)):
            idx = index_generator.pop()
            data.append(dataset[idx])

        return _collate_fn(data)
    
# Load data
train = PneumoniaMNIST(split="train", download=True, size=128)
test = PneumoniaMNIST(split="test", download=True, size=128)
val = PneumoniaMNIST(split="val", download=True, size=128)

train_load = np.array(get_loader(train, True))
test_load = np.array(get_loader(test, True))
val_load = np.array(get_loader(val, False))

# Noise up
noise_factor = 0.2

train_noisy = train_load + noise_factor * tf.random.normal(shape=train_load.shape)
test_noisy = test_load + noise_factor * tf.random.normal(shape=test_load.shape)
val_noisy = val_load + noise_factor * tf.random.normal(shape=val_load.shape)

train_noisy = tf.clip_by_value(train_noisy, clip_value_min=0., clip_value_max=1.)
test_noisy = tf.clip_by_value(test_noisy, clip_value_min=0., clip_value_max=1.)
val_noisy = tf.clip_by_value(val_noisy, clip_value_min=0., clip_value_max=1.)

# Model set up functions
def setup_model():
    n = 128
    chan = 1
    input_img = Input(shape=(n, n, chan))

    l1 = Conv2D(32, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input_img)
    l2 = Conv2D(32, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l1)
    l3 = MaxPooling2D(padding='same')(l2)

    l4 = Conv2D(64, (3, 3),  padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l3)
    l5 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l4)
    l6 = MaxPooling2D(padding='same')(l5)

    # Decoder

    l8 = UpSampling2D()(l6)
    l9 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10),)(l8)
    l10 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l9)
    l11 = add([l5, l10])

    l12 = UpSampling2D()(l11)
    l13 = Conv2D(32, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l12)
    l14 = Conv2D(32, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l13)
    l15 = add([l14, l2])

    # chan = 3, for RGB
    decoded = Conv2D(chan, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(l15)

    return Model(input_img, decoded)

def generate_noise(data, noise_factor = 0.2):
    return data + noise_factor * tf.random.normal(shape=data.shape)

def split_data(data, n, noise_factor = 0.2):
    m = len(data)
    clean_split = []

    for i in range(n):
        clean_split.append(data[m * i // n: m * (i + 1) // n])

    noisy_split = [generate_noise(x, noise_factor) for x in clean_split]

    return clean_split, noisy_split

def neptune_log(epoch, autoencoder):
    # print("Evaluation")
    run["evaluation/mse"].append(autoencoder.evaluate(val_noisy, val_load))
    run[f"images/reconstructed_{epoch + 1}"].upload(neptune.types.File.as_image(autoencoder.predict(val_noisy)[0]))
    # print("")

def neptune_val_images(autoencoder):
    # print("Final evaluation")
    decoded_imgs = autoencoder.predict(val_noisy)
    
    for i in range(1, 6):
        run[f"validation/original_{i}"].upload(neptune.types.File.as_image(val_load[i]))
        run[f"validation/noisy_{i}"].upload(neptune.types.File.as_image(val_noisy[i]))
        run[f"validation/reconstructed_{i}"].upload(neptune.types.File.as_image(decoded_imgs[i]))

def run_model(n, epochs):
    # Make overall model
    autoencoder = setup_model()
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=losses.MeanSquaredError())

    # Log validation images
    run["images/original"].upload(neptune.types.File.as_image(val_load[0]))
    run["images/noisy"].upload(neptune.types.File.as_image(val_noisy[0]))

    # Make n models
    models = [setup_model() for _ in range(n)]
    for model in models:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=losses.MeanSquaredError())
    
    # Split data
    train, train_noisy = split_data(train_load, n)
    test, test_noisy = split_data(test_load, n)

    # Get central weights
    primary_weights = autoencoder.get_weights()
    for model in models:
        model.set_weights(primary_weights)
    
    # Train the networks
    for epoch in range(epochs):
        # print(f"Epoch {epoch + 1}")

        # Train each model
        for i, model in enumerate(models):
            # print(f"Encoder {i + 1}")
            model.fit(train_noisy[i], train[i], batch_size=32, epochs=1, shuffle=True, validation_data=(test_noisy[i], test[i]))
            run[f"evaluation/encoder{i + 1}/mse"].append(model.evaluate(test_noisy[i], test[i]))

        # Find each updates
        weights = [model.get_weights() for model in models]
        weight_update = [0 for _ in primary_weights]

        # Find the aggregated update
        for weight in weights:
            weight_update = [wu + (w - w0) / n for wu, w, w0 in zip(weight_update, weight, primary_weights)]

        # Update the primary weights
        primary_weights = [w0 + wu for w0, wu in zip(primary_weights, weight_update)]
        
        # Set the primary weights
        for model in models:
            model.set_weights(primary_weights)
        
        # print(f"Epoch {epoch + 1} completed")
        # print("")
        
        # Log the results
        autoencoder.set_weights(primary_weights)
        neptune_log(epoch, autoencoder)
    
    neptune_val_images(autoencoder)

    autoencoder.save_weights("./Checkpoints/model_full.weights.h5")
    
    return autoencoder

# Run the model
n = input("Number of clients: ")

autoencoder = run_model(int(n), 100)