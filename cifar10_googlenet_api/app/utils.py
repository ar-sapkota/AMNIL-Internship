# utils.py contains small helper functions to make fastapi app clean. 
# read uploaded image from fastapi requests
# converts into PIL images, and passed to predict()

from PIL import Image
from fastapi import UploadFile

def read_image(file: UploadFile) -> Image.Image:
    '''
    reads an uploaded fastapi file and converts into pil image

    args: 
        file(UploadFile): file uploaded via fastapi
    
    returns:
        PIL.Image.Image: RGB image
    '''

    image = Image.open(file.file).convert("RGB")
    return image