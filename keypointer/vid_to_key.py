#! /usr/bin/env python
'''This script allows to extract a sequence of person keypoints from a video
and stores these as a numpy array.'''

import cv2
import json
import numpy as np
import os
from tqdm import tqdm, trange


PATH_TO_OPENPOSE = '/export/home/jhaux/NIPS18/Patrick/evaluations/openpose'

N_PERSON = 1

VIDEO_TYPES = ['mp4', 'avi', 'mov', 'mpeg']
IMAGE_TYPES = ['jpg', 'jpeg', 'png']


def istype(filename, types):
    '''Returns True if ``filename`` has an ending, which is one of
    ``types``.'''

    if not os.path.isdir(filename):
        ending = filename.slpit('.')[-1]
        if ending is in types:
            return True
    return False


def isimage(filename):
    '''See :func:`istype`. Tests for {}.'''.format(IMAGE_TYPES)
    return istype(filename, IMAGE_TYPES)


def isvideo(filename):
    '''See :func:`istype`. Tests for {}.'''.format(VIDEO_TYPES)
    return istype(filename, VIDEO_TYPES)


def video_to_keypoints(video_file
                       clean_json=True,
                       clean_images=False):
    '''calculates Person keypoints per frame for a given video.

    Args:
        video_file (str): Path/to/the/video to convert. If this path leads to
            a folder, either all videos in the folder are converted or each
            image in the folder is interpreted as a frame in one video.
        clean_json (bool): Delete generated json files
        clean_images (bool): Delete generated image files
    '''

    if isvideo(video_file):
        path_to_frames = video_to_images(video_file)
    elif os.path.isdir(video_file):
        files = os.listdir(video_file)

        has_images = any([isimage(f) for f in files])
        has_videos = any([isvideo(f) for f in files])
        assert (has_images and not has_videos) \
                or (not has_images and has_videos), \
                raise ValueError('The directory at video_file should either '
                    'only contain videos or images, but not both.')

        if has_images:
            path_to_frames = video_file
        if has_videos:
            videos = filter(lambda f: isvideo(f), files)
            for video in tqdm(videos, descr='video'):
                kp, c = video_to_keypoints(os.path.join(root, video))
    else:
        raise ValueError('video_file need to be a path leading to a video '
                + '(file ending with {}) '.format(VIDEO_TYPES)
                + 'or leading to a folder full of images/videos, '
                + 'but is {}'.format(video_file))

    kp_path, conf_path = keypoints_from_images(path_to_frames)

    return kp_path, conf_path


def video_to_images(path_to_video):
    '''Converts a video to a set of images, stored in a subdirectory of the
    location of the video.

    Args:
        path_to_video (str): Where to find the video.

    Returns:
        str: Path to the subdirectory containing the frames of the video.
    '''

    # Make a directory where to store the frames
    parent_dir, video_name = os.path.split(path_to_video)
    video_name = video_name.split('.')[0]
    image_dir = os.path.join(parent_dir, video_name+'_frames')

    os.mkdir(image_dir)

    video = cv2.VideoCapture(path_to_video)

    number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame in trange(number_of_frames, descr='frame'):
        success, image = video.read()
        if not success:
            break
        cv2.imwrite('{}_{}.jpg'.format(frame, video_name))

    print('Converted {} to {}/{} frames stored at {}'
            .format(video_name, frame, number_of_frames, image_dir))

    return image_dir


def keypoints_from_images(path_to_frames):
    '''For each image file at the given path finds person keypoints.

    Args:
        path_to_frames (str): Where to find the images. Images should
            follow the naming convention ``FRAME_VIDEONAME.jpg``.

    Returns:
        str: path to the keypoint file, with keypoints for each image.
        str: path to the confidence file, with confidences for each keypoint in
            each frame.
    '''

    path_of_json_files = calculateKeypoints(path_to_frames)
    
    jsons = os.listdir(path_of_json_files)
    jsons = sorted(jsons, key=lambda n: int(n.split('_')[0]))

    all_keypoints = []
    confidences = []
    for json_frame in tqdm(jsons, descr='reading json'):
        with open(json_frame, 'r') as kp_file:
            keypoints = json.loads(kp_file.read())

            people = keypoints['people']
            if len(people) > 0:
                person_keypoints = []
                person_confidences = []
                for person in people:
                    keypoints = person['pose_keypoints_2d']
                    keypoints = np.array(keypoints).reshape([18, 3])
                    confidence = keypoints[:, 2]
                    keypoints = keypoints[:, :2]

                    person_keypoints += [keypoints]
                    person_confidences += [confidence]

                person_keypoints = np.stack(person_keypoints, axis=0)
                person_keypoints = np.squeeze(person_keypoints, axis=0)
                person_confidences = np.stack(person_confidences, axis=0)
                person_confidences = np.squeeze(person_confidences, axis=0)

                all_keypoints += [person_keypoints]
                confidences += [person_confidences]

    all_keypoints = np.stack(all_keypoints, axis=0)
    confidences = np.stack(confidences, axis=0)

    keypoint_file = os.path.join(path_to_frames, 'keypoints.npy')
    confidence_file = os.path.join(path_to_frames, 'confidences.npy')

    np.save(keypoint_file, all_keypoints)
    np.save(confidence_file, confidences)

    return keypoint_file, confidence_file


def calculateKeypoints(root,
                       cvd,
                       pto=PATH_TO_OPENPOSE,
                       n_persons=N_PERSON,
                       write_json='keypoints'):
    '''Runs openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)
    on the images in root and stores a json for each image.
    
    Args:
        root (str): Path to image folder
        cvd (int or None): Index of gpu to use. Can be anything that
            works with CUDA_VISIBLE_DEVICES and is a single device! Set to \'\'
            if None.
        pto (str): Path to openpose binary. There is an installation on the
            hci servers, but you can also supply your own.
            Defaults to {pto}.
        n_persons (int): Maximum number of persons to detect in each image.
        write_json (str): path where the json files are stored. If relative
            is a subdirectory of root.

    Returns:
        str: Path to the json files
    '''.format(pto=PATH_TO_OPENPOSE)

    # Interpret gpu device
    cvd = '' if cvd is None else str(cvd)

    json_path = os.path.join(root, write_json)

    # run openpose
    os.system('cd {};'.format(pto)
              + 'CUDA_VISIBLE_DEVICES={} '.format(cvd)
              + '   ./build/examples/openpose/openpose.bin '
              + '   -image_dir {} '.format(root)
              + '   -write_json {} '.format(json_path)
              + '   -keypoint_scale 3'  # [-1, 1]
              + '   -number_people_max {}'.format(n_persons)
              + '   -num_gpu 1'
              + '   -display 0'
              + '   -render_pose 1')

    return json_path


if __name__ == '__main__':
    from argparse import ArgumentParser

    P = argparse.ArgumentParser()

    P.add_argument('--v', type=str, default='.',
                   help='Path to the video(s) or the images to parse.')
    P.add_argument('--ci', action='store_true', default=False,
                   help='If this flag is set, all generated images will be '
                        'deleted after the keypoints are generated.')
    P.add_argument('--cj', action='store_true', default=False,
                   help='If this flag is set, all generated json files will '
                        'be deleted after the keypoints are generated.')
    P.add_argument('--np', type=int, default=1,
                   help='Maximum number of persons to detect in the video(s)')

    args = P.parse_args()

    vp = args.v
    clear_i = args.ci
    clear_j = args.cj

    kp_path, c_path = video_to_keypoints(vp, clear_j, clear_i)
    print('Generated keypoints can be found at {}.'.format(kp_path))
