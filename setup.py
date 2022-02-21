from setuptools import setup
from setuptools import find_packages

setup(

    name='DeepAgent',
    version='1.0',
    description='A Deep Reinforcement Learning Framework',
    author='Wuwei Zhang',
    author_email='sgwzha23@liverpool.ac.uk',
    url='https://github.com/LANNDS18/DeepRL_DemonAttack',

    python_requires='>=3.8',
    install_requires=[
        'termcolor',
        'pandas',
        'jupyter',
        'matplotlib',
        'tensorflow-macos',
        'gym',
        'numpy',
        'opencv-python'
    ],
    packages=find_packages()
)
