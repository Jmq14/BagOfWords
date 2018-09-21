import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words


def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")

    image_names = train_data['image_names']
    labels = train_data['labels']
    SPM_layer_num = 2
    features = []
    for i, img_path in enumerate(image_names):
        print(i)
        features.append(
            get_image_feature("../data/"+img_path[0], dictionary, SPM_layer_num, dictionary.shape[0]))

    np.savez('trained_system.npz', dictionary=dictionary,
             features=np.array(features),
             labels=labels,
             SPM_layer_num=SPM_layer_num)


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''
    test_data = np.load("../data/test_data.npz")
    test_labels = test_data['labels']

    trained_system = np.load("trained_system.npz")
    features = trained_system['features']
    labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    SPM_layer_num = trained_system['SPM_layer_num']

    confusion = np.zeros((8, 8))

    for i, img_path in enumerate(test_data['image_names']):
        print(i)
        f = get_image_feature("../data/" + img_path[0], dictionary, SPM_layer_num, dictionary.shape[0])
        print(f)
        print(f.shape)
        intersection = distance_to_set(f, features)
        index = np.argmax(intersection)
        predicted_label = labels[index]
        true_label = test_labels[i]
        confusion[true_label, predicted_label] += 1

    return confusion, np.trace(confusion) / np.sum(confusion)


def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    image = imageio.imread(file_path)
    wordmap = visual_words.get_visual_words(image, dictionary)
    return get_feature_from_wordmap_SPM(wordmap, layer_num, K)


def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    minimum = np.minimum(word_hist, histograms)
    intersection = np.sum(minimum, axis=1)
    return intersection


def get_feature_from_wordmap(wordmap, dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    hist, _ = np.histogram(wordmap, bins=range(dict_size+1))
    return hist


def get_feature_from_wordmap_SPM(wordmap, layer_num, dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    h = wordmap.shape[0]
    w = wordmap.shape[1]
    hist_all = []

    # the finest layer
    n = np.power(2, layer_num)
    nh = int(h / n)
    nw = int(w / n)
    weight = 1/2.
    for i in range(n):
        for j in range(n):
            hist = get_feature_from_wordmap(
                wordmap[i*nh:(i+1)*nh, j*nw:(j+1)*nw], dict_size)
            hist_all.append(hist / (nh*n*nw*n) * weight)

    # aggregates
    start = 0
    for layer in range(layer_num)[::-1]:
        if layer != 0:
            weight = 1/2.
        else: weight = 1.
        n = np.power(2, layer)
        for i in range(n):
            for j in range(n):
                index = start + i*n*4 + j*2
                tmp_hist = weight * (hist_all[index] + hist_all[index+1] +
                                     hist_all[index+n*2] + hist_all[index+n*2+1])
                hist_all.append(tmp_hist)
        start += (n * 2) ** 2
    return np.concatenate(hist_all)
