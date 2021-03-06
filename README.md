KeyPointer
==========

Keypoint extractor using openpose and opencv.

Given a video generates a numpy array containing the extracted keypoints and
cleans up afterweards, if wished.

Installation:
-------------

Navigate to this repository.
This will install the tool `vid_to_key` and link it to `$HOME/.local/bin`:
```
pip install -e .
```
Add ``-vvv`` flag to see all installation outputs.

If you cannot find the executable `vid_to_key` afterwards and/or adding
``-vvv`` does not produce an output line ``linking script``, try the following:
```
python setup.py install
```

Now it can be called from anywhere on you system. If you installed it in a
virtualenv, make sure to start the virtualenv before calling the script.

Example usage:
--------------

**_If run on hci servers needs to be run on hcigpu01!_**

This will extract keypoints from all videos or images inside the folder using
the gpu with index 1:
```
cd /path/to/where_my_video_is/
vid_to_key -cvd 1
```
It is equivalent to:
```
vid_to_key --v '/path/to/where_my_video_is/'
```

If fthe folder contains videos, keypoints are extracted for each video.
If it contains images, these are interpreted as coming from one video.
Make sure the images follow the nameing convention `FRAMENUMBER_VIDEONAME.jpg`,
though only the framenumber and the `_` are important.

Note that if the folder contains both videos and images an error is raised.

You can also point to a video directly:
```
vid_to_key --v /path/to/video.mpg
```

The script assumes that there is only one person in the video. This number
can be increased with the `--np` argument:
```
vid_to_key --np 3 --v /path/to/video.mpg
```

During the creation of the keypoint file a lot of images ans json files are
created. These can be deleted using the `--ci` and `--cj` flags respectiveley.
```
vid_to_key --ci --cj --v /path/to/video.mpg
```
