import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import tensorflow as tf
import yaml


# Functie pentru impartirea setului de date
def split_dataset(dataset_df, split_per, seed=1):
    """Impartirea setului de date in antrenare, validare si testare in mod aleatoriu

    Args:
        dataset_df (pandas.DataFrame): contine caile catre imaginile de input si mastile de segmentare
        split_per (dict): un dictionare de forma {"train": float, "valid": float, "test": float} ce descrie
            procentajele pentru fiecare subset
        seed (int, optional): valoarea seed pentru reproducerea impartirii setului de date. Defaults to 1.
    """
    # se amesteca aleatoriu indecsii DataFrame-ului
    # indexul este un numar (de cele mai multe ori) asociat fiecarui rand
    indices = dataset_df.index.to_numpy()
    total = len(indices)
    random.seed(seed)
    random.shuffle(indices)

    # se impart indecsii in functie de procentele primite ca input
    train_idx = int(total * split_per["train"])
    valid_idx = train_idx + int(total * split_per["valid"])
    test_idx = train_idx + valid_idx + int(total * split_per["test"])

    train_indices = indices[:train_idx]
    valid_indices = indices[train_idx:valid_idx]
    test_indices = indices[valid_idx:test_idx]

    #     print(len(train_indices), len(valid_indices), len(test_indices))

    # se adauga o noua coloana la DataFrame care specifica in ce subset
    # face parte o imagine si mastile de segmentare asociate
    dataset_df['subset'] = ""
    dataset_df.loc[train_indices, "subset"] = "train"
    dataset_df.loc[valid_indices, "subset"] = "valid"
    dataset_df.loc[test_indices, "subset"] = "test"

    return dataset_df


def create_dataset_csv(images_dir, right_masks_dir, left_masks_dir, csv_path):
    """Generare csv cu caile catre imaginile de input si mastile de segmentare

    Args:
        images_dir (str or pathlib.Path): calea catre directorul cu imagini de input
        right_masks_dir (str or pathlib.Path): calea catre directorul cu masti de segmentare pt plamanul drept
        left_masks_dir (str or pathlib.Path): calea catre directorul cu masti de segmentare pt plamanul stang
        csv_path (str or pathlib.Path): calea si numele fisierul csv care va fi salvat

    Returns:
        pandas.DataFrame: contine caile catre fiecare imagine de input sau masca de segmentare
    """

    # se citesc toate caile catre imagini si masti de segmentare
    # este important sa fie in aceeasi ordine
    images = sorted(list(pt.Path(images_dir).rglob("*.png")))
    right_masks = sorted(list(pt.Path(right_masks_dir).rglob("*.png")))
    left_masks = sorted(list(pt.Path(left_masks_dir).rglob("*.png")))

    # se verifica daca nu exista masti lipsa pentru unul dintre plamani
    assert len(right_masks) == len(left_masks), \
        f"nr. of right lung masks {len(right_masks)} != {len(left_masks)} nr. of left lung masks"

    # se verifica daca nu exista imagini sau masti lipsa
    assert len(images) == len(left_masks), \
        f"nr. of image{len(images)} != {len(left_masks)} nr. of masks"

    # se creaza un dictionar de liste, pe baza caruia se creaza obiectul de tip pandas.DataFrame
    dataset_data = {"image_path": images, "right_lung_mask_path": right_masks, "left_lung_mask_path": left_masks}

    dataset_df = pd.DataFrame(data=dataset_data)
    dataset_df.to_csv(csv_path, index=False)
    print(f"Saved dataset csv {csv_path}")

    return dataset_df


# Afisarea curbelor de invatare
def plot_acc_loss(result):
    acc = result.history['accuracy']
    loss = result.history['loss']
    val_acc = result.history['val_accuracy']
    val_loss = result.history['val_loss']

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.show()


# Functie - confusion matrix
def coef(y_adev, y_prezis):
    y_adev_1 = tf.reshape(tf.dtypes.cast(y_adev, tf.float32), [-1])
    y_prezis_1 = tf.reshape(tf.dtypes.cast(y_prezis, tf.float32), [-1])
    intersectie = tf.reduce_sum(y_adev_1 * y_prezis_1)
    return  (2. * intersectie + 1.) / (tf.reduce_sum(y_adev_1) + tf.reduce_sum(y_prezis_1) + 1.)
