import nibabel as nib
import numpy as np
from scipy import ndimage
import tensorflow as tf
import random
from nilearn.image import resample_img
from skimage import exposure
from skimage import data, img_as_float
import skimage.transform as skTrans

# Read and load volume
def read_nifti_file(filepath):
    scan = nib.load(filepath)
    # scan = scan.get_fdata()
    return scan


def normalize(volume):
    
    #only for synthesis model training ---
    volume *= 255.0/volume.max() 
    volume = volume.astype("float32")

    #only for style transfer ---
    volume *= 1.0/volume.max() 
    volume = volume.astype("float32") #----
    
    return volume

def resize_volume(volume):
    volume = volume.get_fdata()
    #Optional steps
    # volume = resample_img(volume, target_affine=np.eye(3)*4., interpolation='nearest') #Downsampling
    # volume = resample_img(volume, target_affine=np.eye(3)*0.5, interpolation='nearest') #Upsampling
    # volume = skTrans.resize(volume,(64,64,64),order=1,preserve_range=True)
    # volume = resample_img(img,target_affine=img.affine,target_shape=(64,64,64))
    return volume

def process_scan(path):
    volume = read_nifti_file(path) #read file
    # volume = volume[:,:,80:100] #select the middle 20 slices (optional)
    volume = resize_volume(volume)   
      
    volume = normalize(volume) #normalize
    
    #Change to numpy array
    volume = np.array(volume)
    volume = np.squeeze(volume)
    
    volume = tf.expand_dims(volume, axis=3)
    
    # print(volume.shape)

    return volume

#Data augmentation
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-3, -2, -1, 1, 2, 3]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    
    # Contrast and brightness stretching
    volume = tf.image.random_brightness(volume, 0.03)
    volume = tf.image.random_contrast(volume, 0.5, 0.8)

    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label