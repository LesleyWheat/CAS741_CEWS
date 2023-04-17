# Description
# Retrieves and formats the datasets under tests/datasets
# Used for test cases

# Import libraries
import os
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# Functions


def get_datasetPath():
    # Get local path
    dir_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "datasets")
    return dir_path


def dataset_irisBezdek():
    # UCI MLR: Iris Bezdek
    filepath = os.path.join(get_datasetPath(), "bezdekIris", "bezdekIris.data")
    D, C = get_iris(filepath)
    return (D, C)


def dataset_iris():
    # UCI MLR: Iris plants
    filepath = os.path.join(get_datasetPath(), "iris", "iris.data")
    D, C = get_iris(filepath)
    return (D, C)


def get_iris(filepath):
    # Format Iris datasets

    # Read data
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 0:4]
    data = data.to_numpy()
    labels_temp = df.iloc[:, 4]
    n = (data.shape[0])

    # Remove class names
    labels = np.empty((n), dtype=int)
    classList = labels_temp.unique()
    classes = classList.shape[0]

    for i in range(0, n):
        for j in range(0, classes):
            if (labels_temp[i] == classList[j]):
                labels[i] = j + 1

    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    data_normed = 2* (data - dmin) / (dmax-dmin) -1

    return (data_normed, labels)


def dataset_yeast():
    # UCI MLR: Yeast

    filepath = os.path.join(get_datasetPath(), "yeast", "yeast.data")
    # Read data
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None, delim_whitespace=True)

    # Convert to numpy arrays
    data = df.iloc[:, 1:9]
    data = data.to_numpy()
    labels_temp = df.iloc[:, 9]
    n = (data.shape[0])

    # Remove class names
    labels = np.empty((n), dtype=int)
    classList = labels_temp.unique()
    classes = classList.shape[0]

    for i in range(0, n):
        for j in range(0, classes):
            if (labels_temp[i] == classList[j]):
                labels[i] = j + 1

    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    data_normed = 2* (data - dmin) / (dmax-dmin) -1

    return (data_normed, labels)


def dataset_breastCancer():
    # UCI MLR: breast cancer

    filepath = os.path.join(
        get_datasetPath(), "breast-cancer", "breast-cancer-wisconsin.data")
    # Read data
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 1:10]
    data = data.to_numpy()
    labels = df.iloc[:, 10]
    labels = labels.to_numpy()
    labels = labels.astype(np.int8)

    # Clean data
    i = 0
    while (i < data.shape[0]):
        deleteRow = False
        for j in range(0, data.shape[1]):
            if data[i, j] == "?":
                deleteRow = True
                break

        if (deleteRow):
            data = np.delete(data, i, 0)
            labels = np.delete(labels, i, 0)
        else:
            i += 1

    data = data.astype(np.float16)

    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    data_normed = 2* (data - dmin) / (dmax-dmin) -1

    return (data_normed, labels)


def dataset_wine():
    # UCI MLR: Wine classification

    filepath = os.path.join(get_datasetPath(), "wine", "wine.data")
    # Read data
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 1:14]
    data = data.to_numpy()
    labels = df.iloc[:, 0]
    labels = labels.to_numpy()
    labels = labels.astype(np.int8)

    # Clean data
    i = 0
    while (i < data.shape[0]):
        deleteRow = False
        for j in range(0, data.shape[1]):
            if data[i, j] == "?":
                deleteRow = True
                break

        if (deleteRow):
            data = np.delete(data, i, 0)
            labels = np.delete(labels, i, 0)
        else:
            i += 1

    data = data.astype(np.float32)

    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    data_normed = 2* (data - dmin) / (dmax-dmin) -1

    return (data_normed, labels)


def dataset_glass():
    # UCI MLR: Glass

    # Read data
    filepath = os.path.join(get_datasetPath(), "glass", "glass.data")
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 1:10]
    data = data.to_numpy(dtype=np.float32)
    labels = df.iloc[:, 10]
    labels = labels.to_numpy(dtype=np.int8)

    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    data_normed = 2* (data - dmin) / (dmax-dmin) -1

    return (data_normed, labels)


def dataset_ionosphere():
    # UCI MLR: Ionosphere

    # Read data
    filepath = os.path.join(get_datasetPath(), "ionosphere", "ionosphere.data")
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 0:34]
    data = data.to_numpy()
    labels_temp = df.iloc[:, 34]
    n = (data.shape[0])

    # Remove class names
    labels = np.empty((n), dtype=int)
    classList = labels_temp.unique()
    classes = classList.shape[0]

    for i in range(0, n):
        for j in range(0, classes):
            if (labels_temp[i] == classList[j]):
                labels[i] = j + 1


    return (data, labels)


def dataset_haberman():
    # UCI MLR: Haberman

    # Read data
    filepath = os.path.join(get_datasetPath(), "haberman", "haberman.data")
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 0:3]
    data = data.to_numpy(dtype=np.float32)
    labels = df.iloc[:, 3]
    labels = labels.to_numpy(dtype=np.int8)

    dmax = data.max(axis=0)
    dmin = data.min(axis=0)
    data_normed = 2* (data - dmin) / (dmax-dmin) -1

    return (data_normed, labels)


def dataset_imgSeg():
    # UCI MLR: Image segmentation

    # Read data
    filepath = os.path.join(
        get_datasetPath(), "image_seg", "segmentation.data")
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None)

    # Convert to numpy arrays
    data = df.iloc[:, 1:20]
    data = data.to_numpy()
    labels_temp = df.iloc[:, 0]
    n = (data.shape[0])

    # Remove class names
    labels = np.empty((n), dtype=int)
    classList = labels_temp.unique()
    classes = classList.shape[0]

    for i in range(0, n):
        for j in range(0, classes):
            if (labels_temp[i] == classList[j]):
                labels[i] = j + 1

    return (data, labels)


def dataset_ruspini():
    # Ruspini

    # Read data
    filepath = os.path.join(get_datasetPath(), "ruspini", "ruspini.txt")
    with open(filepath, "r") as csv_file:
        df = pd.read_csv(csv_file, header=None, delim_whitespace=True)

    data = df.iloc[:, 0:2]
    data = data.to_numpy(dtype=np.int16)

    return (data)

print(dataset_iris())