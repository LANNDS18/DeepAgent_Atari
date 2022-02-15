from setuptools import setup
from setuptools import find_packages

setup(

    name='DeepRL',
    version='1.0',
    description='A Deep Reinforcement Learning Framework',
    author='Wuwei Zhang',
    author_email='sgwzha23@liverpool.ac.uk',
    url='',

    python_requires='>=3.8',
    install_requires=[
        'tensorflow-metal',
        'termcolor',
        'pandas',
        'tensorflow' or 'tensorflow-macos',
        'scipy',
        'gym',
        'numpy',
        'matplotlib',
        'opencv-python'
    ],
    packages=find_packages()
)
