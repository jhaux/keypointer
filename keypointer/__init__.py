import os


cwd = os.path.dirname(__file__)
cudnn_script = os.path.join(cwd, 'activate.sh')
os.system('bash \"{}\"'.format(cudnn_script))

os.system('echo $LD_LIBRARY_PATH')
os.system('echo $CUDA_HOME')
print('initialized cudnn using {}'.format(cudnn_script))
