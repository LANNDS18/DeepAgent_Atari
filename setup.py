from setuptools import setup
from setuptools import find_packages

setup(

    name='DeepAgent',
    version='1.0',
    description='A Deep Reinforcement Learning Framework',
    author='Wuwei Zhang',
    author_email='sgwzha23@liverpool.ac.uk',
    url='https://github.com/LANNDS18/DeepAgent_Atari',

    python_requires='>=3.8',
    install_requires=[
        'gym',
        'tensorflow-macos',
        'pandas',
        'numpy',
        'opencv-python',
        'matplotlib',
        'pyglet',
        'termcolor',
    ],
    packages=find_packages()
)
