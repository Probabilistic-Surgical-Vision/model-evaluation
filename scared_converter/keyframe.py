import os
import os.path
from typing import Dict, Optional, Tuple

import cv2

import numpy as np
from numpy import ndarray

import tifffile

import json

ImageSize = Tuple[int, int]
Parameters = Dict[str, ndarray]
RectifyMaps = Tuple[2 * (ndarray,)]


class SCAREDKeyframeConverter:

    CAMERA_PARAM_KEYS = ('R', 'T', 'M1', 'M2', 'D1', 'D2')
    VIDEO_PATH = os.path.join('data', 'rgb.mp4')
    CAMERA_PARAMETERS = 'endoscope_calibration.yaml'
    LEFT_KEYFRAME = 'Left_Image.png'
    RIGHT_KEYFRAME = 'Right_Image.png'
    LEFT_KF_DEPTH = 'left_depth_map.tiff'
    RIGHT_KF_DEPTH = 'right_depth_map.tiff'
    LEFT_IMAGES = 'left'
    RIGHT_IMAGES = 'right'
    SIMPLE_PARAMETERS = 'parameters.json'

    def __init__(self, source: str, target: Optional[str] = None,
                 image_size: Tuple[int, int] = (512, 256),
                 rectify: bool = True) -> None:

        if target is None:
            target = source

        if source == target:
            print('Source directory and target directory are the same. '
                  'Some files will be overwritten.')

        self.source = source
        self.target = target

        self.image_size = image_size

        self.rectify = rectify

    def load_camera_parameters(self) -> Parameters:
        camera_path = os.path.join(self.source, self.CAMERA_PARAMETERS)
        camera = cv2.FileStorage(camera_path, cv2.FileStorage_READ)

        parameters = {}

        for key in self.CAMERA_PARAM_KEYS:
            parameter = camera.getNode(key).mat()
            parameters[key] = np.array(parameter, dtype=float)

        return parameters

    def get_rectify_parameters(self, camera: Parameters,
                               size: ImageSize) -> Parameters:

        r, t = camera['R'], np.squeeze(camera['T'])

        m1, d1 = camera['M1'], np.squeeze(camera['D1'])
        m2, d2 = camera['M2'], np.squeeze(camera['D2'])

        rectify = cv2.stereoRectify(m1, d1, m2, d2, size, r, t, alpha=0,
                                    flags=cv2.CALIB_ZERO_DISPARITY)

        r1, r2, p1, p2, *_ = rectify

        return {'R1': r1, 'R2': r2, 'P1': p1, 'P2': p2}

    def get_rectify_maps(self, camera: Parameters, rectify: Parameters,
                         size: ImageSize, m1type: int = cv2.CV_32FC1) -> dict:

        m1, d1 = camera['M1'], camera['D1']
        m2, d2 = camera['M2'], camera['D2']

        r1, p1 = rectify['R1'], rectify['P1']
        r2, p2 = rectify['R2'], rectify['P2']

        left = cv2.initUndistortRectifyMap(m1, d1, r1, p1, size, m1type)
        right = cv2.initUndistortRectifyMap(m2, d2, r2, p2, size, m1type)

        return {'left': left, 'right': right}

    def get_baseline(self, camera: Parameters) -> float:
        return np.linalg.norm(camera['T'])

    def get_focal_length(self, rectify: Parameters) -> float:
        return rectify['P1'][0, 0]

    def split_image(self, stereo_image: ndarray) -> RectifyMaps:
        height = stereo_image.shape[0]
        return stereo_image[:height//2], stereo_image[height//2:]

    def rectify_image(self, image: ndarray, maps: RectifyMaps) -> ndarray:
        return cv2.remap(image, *maps, cv2.INTER_LINEAR)

    def rectify_depth(self, depth: ndarray, maps: RectifyMaps) -> ndarray:
        return cv2.remap(depth[:, :, 2:3], *maps, cv2.INTER_LINEAR)

    def save_baseline_and_focal_length(self, camera: Parameters,
                                       rectify: Parameters) -> None:
        parameters = {
            'focal': self.get_focal_length(rectify),
            'baseline': self.get_baseline(camera)
        }

        parameters_path = os.path.join(self.target, self.SIMPLE_PARAMETERS)

        with open(parameters_path, 'w+') as f:
            json.dump(parameters, f, indent=4)

    def save_video_images(self, left_maps: RectifyMaps,
                          right_maps: RectifyMaps) -> None:

        video_path = os.path.join(self.source, self.VIDEO_PATH)

        left_target_dir = os.path.join(self.target, self.LEFT_IMAGES)
        right_target_dir = os.path.join(self.target, self.RIGHT_IMAGES)

        os.makedirs(left_target_dir, exist_ok=True)
        os.makedirs(right_target_dir, exist_ok=True)

        video = cv2.VideoCapture(video_path)

        i = 0

        while video.isOpened():
            continue_flag, image = video.read()

            if not continue_flag:
                break

            left, right = self.split_image(image)

            if self.rectify:
                left = self.rectify_image(left, left_maps)
                right = self.rectify_image(right, right_maps)

            left_target = os.path.join(left_target_dir, f'image_{i:06}.png')
            right_target = os.path.join(right_target_dir, f'image_{i:06}.png')

            left = cv2.resize(left, self.image_size,
                              interpolation=cv2.INTER_LINEAR)
            right = cv2.resize(right, self.image_size,
                               interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(left_target, left, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(right_target, right, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            i += 1

    def save_keyframes(self, left_maps: RectifyMaps,
                       right_maps: RectifyMaps) -> None:

        left_source = os.path.join(self.source, self.LEFT_KEYFRAME)
        right_source = os.path.join(self.source, self.RIGHT_KEYFRAME)

        left_target = os.path.join(self.target, self.LEFT_KEYFRAME)
        right_target = os.path.join(self.target, self.RIGHT_KEYFRAME)

        left = cv2.imread(left_source)
        right = cv2.imread(right_source)

        if self.rectify:
            left = self.rectify_image(left, left_maps)
            right = self.rectify_image(right, right_maps)

        cv2.imwrite(left_target, left, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(right_target, right, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def save_depth_maps(self, left_maps: RectifyMaps,
                        right_maps: RectifyMaps) -> None:

        left_source = os.path.join(self.source, self.LEFT_KF_DEPTH)
        right_source = os.path.join(self.source, self.RIGHT_KF_DEPTH)

        left_target = os.path.join(self.target, self.LEFT_KF_DEPTH)
        right_target = os.path.join(self.target, self.RIGHT_KF_DEPTH)

        left = tifffile.imread(left_source)
        right = tifffile.imread(right_source)

        if self.rectify:
            left = self.rectify_depth(left, left_maps)
            right = self.rectify_depth(right, right_maps)

        tifffile.imwrite(left_target, left)
        tifffile.imwrite(right_target, right)

    def extract(self) -> None:
        camera = self.load_camera_parameters()

        left_keyframe_path = os.path.join(self.source, self.LEFT_KEYFRAME)
        left_keyframe = cv2.imread(left_keyframe_path)

        height, width, _ = left_keyframe.shape
        image_size = (width, height)

        rectify = self.get_rectify_parameters(camera, image_size)
        maps = self.get_rectify_maps(camera, rectify, image_size)

        os.makedirs(self.target, exist_ok=True)

        self.save_baseline_and_focal_length(camera, rectify)

        left_maps = maps['left']
        right_maps = maps['right']

        self.save_keyframes(left_maps, right_maps)
        self.save_depth_maps(left_maps, right_maps)
        self.save_video_images(left_maps, right_maps)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('source', type=str,
                        help='The path to the original SCARED keyframe.')
    parser.add_argument('--target', '-t', default=None, type=str,
                        help='The path to save the converted keyframe to.')
    parser.add_argument('--rectify', default=False, action='store_true',
                        help='Rectify the images when saving.')
    parser.add_argument('--image-size', type=int, nargs=2, default=(512, 256),
                    help='The size to make all video images.')

    args = parser.parse_args()

    converter = SCAREDKeyframeConverter(**vars(args))
    converter.extract()
