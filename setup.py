from setuptools import setup, find_packages

setup(
    name='raft_stereo',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'matplotlib',
        'tensorboard',
        'scipy',
        'opt_einsum',
        'glob',
        'skimage',
        'Pillow',
        'imageio',
        'numpy'
    ]
)

