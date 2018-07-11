#! /usr/bin/env python
'''This script allows to extract a sequence of person keypoints from a video
and stores these as a numpy array.'''

import cv2
import json
import numpy as np
import os
from tqdm import tqdm, trange


PATH_TO_OPENPOSE = '/export/home/jhaux/NIPS18/Patrick/evaluations/openpose'
CVD = 1

N_PERSON = 1

VIDEO_TYPES = ['mp4', 'avi', 'mov', 'mpeg']
IMAGE_TYPES = ['jpg', 'jpeg', 'png']


def istype(filename, types):
    '''Returns True if ``filename`` has an ending, which is one of
    ``types``.'''

    if not os.path.isdir(filename):
        ending = filename.split('.')[-1]
        if ending in types:
            return True
    return False


def isimage(filename):
    '''See :func:`istype`. Tests for {}.'''.format(IMAGE_TYPES)
    return istype(filename, IMAGE_TYPES)


def isvideo(filename):
    '''See :func:`istype`. Tests for {}.'''.format(VIDEO_TYPES)
    return istype(filename, VIDEO_TYPES)


def video_to_keypoints(video_file,
                       clean_json=True,
                       clean_images=False):
    '''calculates Person keypoints per frame for a given video.

    Args:
        video_file (str): Path/to/the/video to convert. If this path leads to
            a folder, either all videos in the folder are converted or each
            image in the folder is interpreted as a frame in one video.
        clean_json (bool): Delete generated json files
        clean_images (bool): Delete generated image files. Ignored if
            video_file is the path to an image directory.
    '''

    if isvideo(video_file):
        path_to_frames = video_to_images(video_file)
    elif os.path.isdir(video_file):
        files = os.listdir(video_file)

        has_images = any([isimage(f) for f in files])
        has_videos = any([isvideo(f) for f in files])
        assert (has_images and not has_videos) \
                or (not has_images and has_videos), \
                'The directory at video_file should either ' \
                    'only contain videos or images, but not both.'

        if has_videos:
            videos = list(filter(lambda f: isvideo(f), files))
            print('Processing the following videos:')
            for v in videos:
                print('-', v)
            keypoint_files = []
            confidence_files = []
            for video in tqdm(videos, desc='video'):
                print('Processing {}'.format(video))
                kp, c, j = video_to_keypoints(os.path.join(video_file, video))
                keypoint_files += [kp]
                confidence_files += [c]

            print('')
            return keypoint_files, confidence_files
        else:
            path_to_frames = os.path.abspath(video_file)
            clean_images = False
    else:
        raise ValueError('video_file need to be a path leading to a video '
                + '(file ending with {}) '.format(VIDEO_TYPES)
                + 'or leading to a folder full of images/videos, '
                + 'but is {}'.format(video_file))

    kp_path, conf_path, json_path = keypoints_from_images(path_to_frames)

    if clean_json:
        clean(json_path, '.json')
    if clean_images:
        clean(path_to_frames, '.jpg')

    return kp_path, conf_path


def clean(directory, ending='.json'):
    '''Removes all json and images files in the given directory.'''
    
    files = os.listdir(directory)
    files_to_clean = filter(lambda f: ending == f.split('.')[-1], files)

    for f in files_to_clean:
        os.remove(os.path.join(directory, f))

    files = os.listdir(directory)
    if len(files) == 0:
        os.rmdir(directory)

def video_to_images(path_to_video):
    '''Converts a video to a set of images, stored in a subdirectory of the
    location of the video.

    Args:
        path_to_video (str): Where to find the video.

    Returns:
        str: Path to the subdirectory containing the frames of the video.
    '''

    # Make a directory where to store the frames
    parent_dir, video_name = os.path.split(os.path.abspath(path_to_video))
    video_name = video_name.split('.')[0]
    image_dir = os.path.join(parent_dir, video_name+'_frames')

    try:
        os.mkdir(image_dir)
    except FileExistsError as e:
        pass

    video = cv2.VideoCapture(path_to_video)

    number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame in trange(number_of_frames, desc='frame'):
        success, image = video.read()
        if not success:
            break
        image_name = '{}_{}.jpg'.format(frame, video_name)
        cv2.imwrite(os.path.join(image_dir, image_name), image)
    print('')

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

    path_of_json_files = calculateKeypoints(path_to_frames, CVD)

    return json_to_numpy(path_to_json_files)
    

def json_to_numpy(path_to_json_files):
    '''Converts a set of sortable json files into one numpy array.
    
    Args:
        path_to_json_files: Directory containing all json files.
            Must be named INDEX_SOMETHING.json 
            Directory must only contain json files.

    Returns:
        str: path to keypoint_file
        str: path to confidence_file
        str: path of json files
    '''
    jsons = os.listdir(path_to_json_files)
    jsons = sorted(jsons, key=lambda n: int(n.split('_')[0]))

    all_keypoints = []
    confidences = []
    missed = []
    for json_frame_ in tqdm(jsons, desc='reading json files'):
        json_frame = os.path.join(path_of_json_files, json_frame_)
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
            else:
                all_keypoints += [np.zeros([18, 3])]
                confidences += [np.zeros([18, 3])]
                missed += [json_frame_]

    if len(missed) > 0:
        print('The following json files contained no keypoints:')
        for m in missed:
            print(m)

    print('')
    all_keypoints = np.stack(all_keypoints, axis=0)
    confidences = np.stack(confidences, axis=0)

    keypoint_file = os.path.join(path_to_frames, 'keypoints.npy')
    confidence_file = os.path.join(path_to_frames, 'confidences.npy')

    np.save(keypoint_file, all_keypoints)
    np.save(confidence_file, confidences)

    return keypoint_file, confidence_file, path_of_json_files


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
    cvd = '0' if cvd is None else str(cvd)
    num_gpu = 1

    json_path = os.path.join(root, write_json)

    print('\nRUNNING OPENPOSE. THIS CAN TAKE A WHILE\n')

    # run openpose
    os.system('cd {};'.format(pto)
              + 'CUDA_VISIBLE_DEVICES={} '.format(cvd)
              + '   ./build/examples/openpose/openpose.bin '
              + '   -image_dir \"{}\" '.format(root)
              + '   -write_json \"{}\" '.format(json_path)
              + '   -keypoint_scale 3'  # [-1, 1]
              + '   -number_people_max {}'.format(n_persons) 
              # + '   -num_gpu {}'.format(num_gpu)
              + '   -display 0'
              # + '   -render_pose 1')
              )

    return json_path


if __name__ == '__main__':
    import keypointer
    from argparse import ArgumentParser

    P = ArgumentParser()

    P.add_argument('--v', type=str, default='.',
                   help='Path to the video(s) or the images to parse.')
    P.add_argument('--ci', action='store_true', default=False,
                   help='If this flag is set, all generated images will be '
                        'deleted after the keypoints are generated.')
    P.add_argument('--cj', action='store_true', default=True,
                   help='If this flag is set, all generated json files will '
                        'be deleted after the keypoints are generated.')
    P.add_argument('--np', type=int, default=1,
                   help='Maximum number of persons to detect in the video(s)')
    P.add_argument('-cvd', type=int, default=None,
                   help='Index of GPU to do the computations on.')

    args = P.parse_args()

    vp = args.v
    clear_i = args.ci
    clear_j = args.cj
    CVD = args.cvd


    kp_path, c_path = video_to_keypoints(vp, clear_j, clear_i)
    print('Generated keypoints can be found at:')
    if isinstance(kp_path, str):
        print('- {}.'.format(kp_path))
    else:
        for kp in kp_path:
            print('- {}.'.format(kp))
