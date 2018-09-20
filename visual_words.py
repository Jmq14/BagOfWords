import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os, time
import matplotlib.pyplot as plt
import util
import random


def preprocess_image(image):
    if np.max(image) > 1:
        image = image.astype('float') / 255
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image[:, :, :3]


def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    def filter_bank_on_image(filter_func, s, order=None):
        filtered_image = []
        for channel in range(3):
            if order:
                filtered_image.append(filter_func(image[:, :, channel], sigma=s, order=order, truncate=2.5))
            else:
                filtered_image.append(filter_func(image[:, :, channel], sigma=s, truncate=2.5))
        # shape in (H, W, 3)
        return np.array(filtered_image).transpose((1, 2, 0))

    image = preprocess_image(image)
    image = skimage.color.rgb2lab(image)
    scales = [1, 2, 4, 8, 8 * np.sqrt(2)]

    responses = []
    for scale in scales:
        # Gaussian filter
        responses.append(filter_bank_on_image(scipy.ndimage.gaussian_filter, scale))
        # Laplacian of Gaussian
        responses.append(filter_bank_on_image(scipy.ndimage.gaussian_laplace, scale))
        # derivative of gaussian
        responses.append(filter_bank_on_image(scipy.ndimage.gaussian_filter, scale, order=[0, 1]))
        responses.append(filter_bank_on_image(scipy.ndimage.gaussian_filter, scale, order=[1, 0]))
    # shape in (H, W, 3F)
    return np.concatenate(responses, axis=2)


def get_visual_words(image, dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    response = extract_filter_responses(image)
    h = response.shape[0]
    w = response.shape[1]
    response = response.reshape(h*w, -1)
    d = scipy.spatial.distance.cdist(response, dictionary, 'euclidean')
    d = d.reshape(h, w, -1)
    wordmap = np.argmin(d, axis=-1)
    return wordmap


def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''

    i, alpha, image_path = args
    if os.path.isfile('../data/tmp/{}.npy'.format(i)): return

    image = imageio.imread(image_path)
    response = extract_filter_responses(image)
    index_x = np.random.choice(image.shape[0], alpha)
    index_y = np.random.choice(image.shape[1], alpha)
    np.save("../data/tmp/{}.npy".format(i), response[index_x, index_y, :])
    return


def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    '''

    train_data = np.load("../data/train_data.npz")
    image_names = train_data['image_names']
    labels = train_data['labels']

    alpha = 50
    K = 300
    if not os.path.exists('../data/tmp'):
        os.mkdir('../data/tmp')

    with multiprocessing.Pool(processes=num_workers) as pool:
        args_list = [(i, alpha, "../data/" + x[0]) for i, x in enumerate(image_names)]
        pool.map(compute_dictionary_one_image, args_list)

    responses = None
    for i in range(image_names.shape[0]):
        if responses is None:
            responses = np.load('../data/tmp/{}.npy'.format(i))
        else:
            responses = np.vstack((responses, np.load('../data/tmp/{}.npy'.format(i))))

    print('begin k-means...')
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(responses)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy', dictionary)
