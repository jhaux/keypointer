from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info


def my_thing():
    '''Make script executable and link it'''
    print('linking script')
    import os

    cwd = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(cwd, 'keypointer', 'vid_to_key.py')

    # Make executable
    os.system('chmod +x \"{}\"'.format(script_path))

    link_path = '$HOME/.local/bin/vid_to_key'

    # Link script somwhere, where PATH is pointing
    os.system('ln -s \"{}\" \"{}\"'.format(script_path, link_path))


class PostInstallCommand(install):
    '''Does not work'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        install.run(self)
        my_thing()


class PostDevelopCommand(develop):
    '''Does not work'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        develop.run(self)
        my_thing()


class PostEggCommand(egg_info):
    '''Does not work'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        egg_info.run(self)
        my_thing()


setup(name='keypointer',
      version='0.1',
      description='Given a video extract keypoints using opencv and openpose.',
      author='Johannes Haux',
      author_email='jo.mobile.2011@gmail.com',
      license='MIT',
      packages=['keypointer'],
      install_requires=[
          'argparse',
          'numpy',
          'opencv-python',
          'tqdm'
      ],
      cmdclass={'install': PostInstallCommand,
                'develop': PostDevelopCommand,
                'egg_info': PostEggCommand})
