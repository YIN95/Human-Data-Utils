from setuptools import setup
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open(path.join(DIR, 'README.md')) as f:
    README = f.read()

setup(
    name='hmdata',
    version='0.0.1',
    packages='hmdata',
    python_requires='>=3.5',
    author='Wenjie Yin',
    author_email='yinwenjie159@hotmail.com',
    url='https://github.com/YIN95/Human-Data-Utils',
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=INSTALL_PACKAGES,
)
