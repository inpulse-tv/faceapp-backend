from os import listdir
import cv2
import numpy as np
from numpy.linalg import norm
from PIL import Image

BASE = "/app/volume"


def most_similar_photo(celebrity_name, user_embed, model):
    """
    This function allows to get the celebrity's most similar photo to the user
    :param celebrity_name: the name of the celebrity.
    :param user_embed: the embedding of the user's image
    :param model: the embedding model (FaceNet)
    :return: Tuple that consists of the celebrity's most similar image to the user and its path
    """

    # get the path of celebrity directory
    celeb_directory = BASE + "/" + celebrity_name

    # get the embeddings of the celebrity's images
    embeddings = dict()
    for file in listdir(celeb_directory):
        img = Image.open(celeb_directory + "/" + file)
        image = get_data(img)
        embed = get_embeddings(model, image)
        embeddings[file] = embed

    # get the image that has the minimal distance to the user's embedding
    mini = float("inf")
    ans = ""
    for name, embed in embeddings.items():
        diff = norm(embed-user_embed)
        if mini > diff:
            mini = diff
            ans = name
    ans2 = Image.open(celeb_directory+"/"+ans)

    return ans2, celeb_directory+"/"+ans


def get_embeddings(model, image):
    """
    Compute the embedding of a given image
    :param model: the embedding model (FaceNet)
    :param image: the image to embed
    :return: the embedding of the image
    """
    samples = np.expand_dims(image, axis=0)
    embedding = model.predict(samples)
    return embedding


def get_data(img):
    """
    preprocess the given image
    :param img: the image
    :return: the preprocessed image
    """
    image = img.convert('RGB')
    image = np.asarray(image, dtype=np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (160, 160))
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return image