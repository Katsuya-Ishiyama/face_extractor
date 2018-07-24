# -*- coding: utf-8 -*-

import pathlib
import cv2


FACE_CASCADE_PATH = ('/home/ishiyama/tensorflow/lib/'
                     'python3.5/site-packages/cv2/data/'
                     'haarcascade_frontalface_default.xml')


def extract_faces_from_image(path, scale_factor, min_neighbors):
    """ extract human faces from an image.

    Arguments
    ---------
    path: str
        a path of an image.
    scale_factor: int
        scaleFactor.
    min_neighbors: tuple[int]
        minNeighbor.

    Returns
    -------
        a list object of Numpy arrays, which are extracted faces.
    """
    image = cv2.imread(str(path))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    face_cordinates = face_cascade.detectMultiScale(image_gray,
                                                    scaleFactor=scale_factor,
                                                    minNeighbors=min_neighbors)
    faces = []
    for (left, top, width, height) in face_cordinates:
        bottom = top + height
        right = left + width
        face = image[top:bottom, left:right]
        faces.append(face)
    return faces


def generate_face_image_path(src_image_path):
    pass


src_dir = pathlib.Path('/home/ishiyama/image_scraper/download/')
member_dirs = [p for p in src_dir.iterdir()]
for _dir in member_dirs:
    for image_path in _dir.iterdir():
        pass
