from setuptools import setup

setup(name='keypointer',
    version='0.1',
    description='Given a video extract keypoints using opencv and openpose.',
    author='Johannes Haux',
    author_email='jo.mobile.2011@gmail.com',
    license='MIT',
    packages=['keypointer'],
    dependencies=[
        'argparse',
        'json',
        'numpy',
        'opencv-python',
        'os',
        'tqdm'
    ],
    zip_safe=False)

import os

# Make executable
os.system('chmod +x vid_to_key.py')

cwd = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(cwd, 'vid_to_key.py')
link_path = '$HOME/.local/bin/vid_to_key'

# Link script somwhere, where PATH is pointing
os.system('ln -s {} {}'.format(script_path, link_path))
