# -*- coding: utf-8 -*-

import argparse
import pathlib
import cv2


def get_commandline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir',
                        type=str,
                        required=True,
                        help='A path of the source directory')
    parser.add_argument('--save-dir',
                        type=str,
                        required=True,
                        help='A path of the directory that resized images will be saved in')
    args = parser.parse_args()
    return args


def resize_image(src_path: pathlib.Path, save_path: pathlib.Path, size: tuple):
    img = cv2.imread(str(src_path))
    resized_img = cv2.resize(img, size)
    cv2.imwrite(str(save_path), resized_img)


def main():
    args = get_commandline_args()
    src_dir = pathlib.Path(args.src_dir)
    save_dir = pathlib.Path(args.save_dir)

    SIZE = (224, 224)

    for d in src_dir.iterdir():
        member_dir = save_dir / d.name
        if not member_dir.exists():
            member_dir.mkdir(parents=True)

        for src_img_path in d.iterdir():
            save_img_path = member_dir / src_img_path.name
            resize_image(
                src_path=src_img_path,
                save_path=save_img_path,
                size=SIZE
            )


if __name__ == '__main__':
    main()

