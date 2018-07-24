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


def generate_face_image_path(src_image_path, face_image_dir, face_id):
    src_image_id = src_image_path.stem
    suffix = src_image_path.suffix
    face_image_id = src_image_id + '{:02d}'.format(face_id)
    face_image_filename = face_image_id + suffix
    _face_image_dir = pathlib.Path(face_image_dir)
    member_id = str(src_image_path.parent).split(',')[-1]
    face_image_path = _face_image_dir.joinpath(member_id, face_image_filename)
    return pathlib.Path(face_image_path)


src_dir = pathlib.Path('/home/ishiyama/image_scraper/download/')
member_dirs = [p for p in src_dir.iterdir()]
for _dir in member_dirs:
    for image_path in _dir.iterdir():
        faces_list = extract_faces_from_image(
            path=image_path,
            scale_factor=1.1,
            min_neighbors=(1, 1)
        )
        for face_id, face in enumerate(faces_list, start=1):
            pass
        #print(image_path)

