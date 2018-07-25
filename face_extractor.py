# -*- coding: utf-8 -*-

import argparse
import logging
import pathlib
import sys
import cv2


FACE_CASCADE_PATH = ('/home/ishiyama/tensorflow/lib/'
                     'python3.5/site-packages/cv2/data/'
                     'haarcascade_frontalface_default.xml')

logger = logging.getLogger(__name__)


def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir',
                        type=str,
                        requierd=True,
                        help='a path to a directory of source images.')
    parser.add_argument('--output-dir',
                        type=str,
                        requierd=True,
                        help='a path to a directory to output images')
    args = parser.parse_args()
    return args


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
    logger.debug('start loading an image: {}'.format(path))
    image = cv2.imread(str(path))
    logger.debug('{} was successfully loaded'.format(path))
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

    if faces:
        faces_num = len(faces)
        logger.debug('{} faces have been detected.'.format(faces_num))
    else:
        logger.warning('No faces have been detected: {}'.format(path))
    return faces


def generate_face_image_path(src_image_path, face_image_dir, face_id):
    src_image_id = src_image_path.stem
    suffix = src_image_path.suffix
    face_image_id = src_image_id + '{:02d}'.format(face_id)
    face_image_filename = face_image_id + suffix
    _face_image_dir = pathlib.Path(face_image_dir)
    member_id = str(src_image_path.parent).split('/')[-1]
    face_image_path = _face_image_dir.joinpath(member_id, face_image_filename)
    return pathlib.Path(face_image_path)


def main():
    file_handler = logging.FileHandler('face_extractor.log')
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] {%(filename)s:%(lineno)s} %(message)s',
        level=logging.INFO,
        handlers=[file_handler]
    )


    args = get_commandline_args()
    src_dir = pathlib.Path(args.src_dir)
    output_dir = pathlib.Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()
        logging.info("Directory {} was created since it doesn't exist")

    member_dirs = [p for p in src_dir.iterdir()]
    for _dir in member_dirs:
        for image_path in _dir.iterdir():
            logging.debug('processing in {}'.format(image_path))
            faces_list = extract_faces_from_image(
                path=str(image_path),
                scale_factor=1.1,
                min_neighbors=1
            )
            for face_id, face in enumerate(faces_list, start=1):
                save_path = generate_face_image_path(
                    src_image_path=image_path,
                    face_image_dir=output_dir,
                    face_id=face_id
                )
                save_path_parent_dir = save_path.parent
                if not save_path_parent_dir.exists():
                    save_path_parent_dir.mkdir()
                    logger.info("Directory {}".format(save_path_parent_dir)
                                " was created since it doesn't exist")
                cv2.imwrite(str(save_path), face)
                logger.debug('{} was successfully saved.'.format(save_path))


if __name__ == '__main__':
    main()

