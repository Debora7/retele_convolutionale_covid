import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import tensorflow as tf
import yaml
import os

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from UNetModel import UNetModel
from PIL import Image, ImageEnhance
from functions import *
from LungSegDataGenerator import *
from datetime import datetime

config = None
with open('config.yaml') as f:  # reads .yml/.yaml files
    config = yaml.safe_load(f)

local_date = datetime.now()

dataset_df = create_dataset_csv(config["data"]["images_dir"],
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])

dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
dataset_df.head(3)

# Vizualizare exemple date folosinf DataGenerator custom
data_gen = LungSegDataGenerator(dataset_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"])
x, y = data_gen[0]
print(x.shape, y.shape)

f, axs = plt.subplots(1, 2)
axs[0].axis('off')
axs[0].set_title("Input")
axs[0].imshow((x[0]*255).astype(np.uint8))


axs[1].axis('off')
axs[1].set_title("Mask")
axs[1].imshow(y[0])
plt.show()

unet = UNetModel()
# n_channels=1, deoarece imaginea de input are un singur canal
# n_classes=1, o singura clasa de prezis -> plaman vs background
unet_model = unet.build(*config["data"]["img_size"], n_channels=3, n_classes=3)
unet_model.summary()

train_df = dataset_df.loc[dataset_df['subset'] == 'train']  # de completat
train_gen = LungSegDataGenerator(train_df,
                                 img_size=config["data"]["img_size"],
                                 batch_size=config["train"]["bs"],
                                 shuffle=True)

valid_df = dataset_df.loc[dataset_df['subset'] == 'valid']  # de completat
valid_gen = LungSegDataGenerator(valid_df,
                                 img_size=config["data"]["img_size"],
                                 batch_size=config["train"]["bs"],
                                 shuffle=True)

# de completat
# se compileaza unet_model cu optimizatorul setat in config.yaml, loss binary crossentropy si metrics accuracy
# se adauga un callback pentru a salva cel mai bun model intr-un fisier .h5
unet_model.compile(loss="binary_crossentropy",
                   optimizer=tf.keras.optimizers.Adam(learning_rate=config['train']['lr']),
                   metrics=["accuracy"])
callbacks = [keras.callbacks.ModelCheckpoint('segmentare.h5', save_best_only=True),
             keras.callbacks.CSVLogger(f'file.csv{local_date}', separator=',', append=False)]
history = unet_model.fit(train_gen,
                         validation_data=valid_gen,
                         epochs=config['train']['epochs'],
                         callbacks=callbacks,
                         workers=1)
unet_model.save('my_model')

plot_acc_loss(history)

# Testarea modelului
test_df = dataset_df.loc[dataset_df['subset'] == 'test']  # de completat
test_gen = LungSegDataGenerator(test_df,
                                img_size=config["data"]["img_size"],
                                batch_size=config["train"]["bs"],
                                shuffle=False)
result = unet_model.evaluate(test_gen)
print(f"Test Acc: {result[1] * 100}")

x, y = test_gen[0]
y_pred = unet_model.predict(x)
y_pred.shape

nr_exs = 4  # nr de exemple de afisat
fig, axs = plt.subplots(nr_exs, 3, figsize=(10, 10))
for i, (img, gt, pred) in enumerate(zip(x[:nr_exs], y[:nr_exs], y_pred[:nr_exs])):
    axs[i][0].axis('off')
    axs[i][0].set_title('Input')
    axs[i][0].imshow(img, cmap='gray')

    axs[i][1].axis('off')
    axs[i][1].set_title('Ground truth')
    axs[i][1].imshow(gt, cmap='gray')

    pred[pred > config['test']['threshold']] = 1
    pred[pred <= config['test']['threshold']] = 0
    pred = pred.astype("uint8")

    axs[i][2].axis('off')
    axs[i][2].set_title('Prediction')
    axs[i][2].imshow(pred, cmap='gray')

    index = coef(gt, pred)
    print(coef)
    axs[i][3].axis('off')
    axs[i][3].set_title('Dice Index')
    axs[i][3].imshow(coef,cmap='gray')

plt.show()
