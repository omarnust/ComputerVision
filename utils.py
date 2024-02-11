import requests  
from io import BytesIO
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Get image given the url
def getImage(url, mode = cv2.IMREAD_COLOR):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, mode)    
    return image
    
def gaussian_noise(image, mean=0, sigma=25):            # most of the time it'll be centered around 0 (you can play with the mean & sigma values to see how it changes)
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = np.clip(image + gauss, 0, 255)              # add the noise to the image and clip it such that no pixel exceeds 255 or is below 0
    return noisy.astype(np.uint8)

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:                       # grayscale -- (height, width)
        black = 0
        white = 255            
    else:                                           # color     -- (height, width, no_channels)
        colorspace = image.shape[2] # no_channels
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')

        else:               # RGBA (A = alpha --> transparency/opacity)
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')

    probs = np.random.random(output.shape[:2])      # generate random probabilities for each pixel
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output

def show_collage(figsize, nrows, ncols, **kwargs):
    plt.figure(figsize=figsize)
    
    if 'images' not in kwargs:
        return False    
    
    n = len(kwargs['images'])
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(kwargs['images'][i], cmap = 'gray')
        plt.title(kwargs['titles'][i])
        plt.axis('off')
    
    # plt will automatically scale in and give a valid image, even if you give it pixel values within -255 and 0 range
    # but OpenCV only works within a valid range (i.e., 0-255, or 0-1)
            