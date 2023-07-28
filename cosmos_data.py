from pathlib import Path
from PIL import Image
import numpy as np

from astropy.io import fits
from tqdm.notebook import tqdm

def read_raw_data(path_to_data='data/COSMOS_23.5_training_sample/'):
    # Path to COSMOS dataset
    catalog_folder = Path(path_to_data)
    # Combine the 57 catalogs into a single array
    galaxy_images = []
    catalogs = catalog_folder.glob("real_galaxy_images_23.5_n*.fits")
    for catalog in tqdm(catalogs):
        for galaxies in fits.open(catalog):
            galaxy_images.append(galaxies.data)
    return galaxy_images


def crop_and_downsample(galaxy_images, n_pixels: int = 32):
    galaxy_images_cropped_downsampled = []

    # Loop over all images, crop and downsample
    for idx, galaxy_image in enumerate(tqdm(galaxy_images)):
        
        im = Image.fromarray(galaxy_image)
        
        ## Cropping
        
        width, height = galaxy_image.shape

        new_dim = np.min([width, height])
        
        # If the image is smaller than the minimum downsampled dimension, skip it
        if new_dim < n_pixels:
            continue

        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = width - np.ceil((width - new_dim) / 2)
        bottom =  height - np.ceil((height - new_dim) / 2)

        # Crop the center of the image
        im = im.crop((left, top, right, bottom))
        
        # Resize/downsample the image
        im = np.array(im.resize((n_pixels, n_pixels)))
        
        # Append cropped and downsampled image to list
        galaxy_images_cropped_downsampled.append(im)
        
    return np.array(galaxy_images_cropped_downsampled)

def read_data(path_to_data, n_pixels=32):
    galaxy_images = read_raw_data(path_to_data)
    return crop_and_downsample(galaxy_images, n_pixels=n_pixels)