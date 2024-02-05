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
    
def gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = image.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
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
    
            