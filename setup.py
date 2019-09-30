import setuptools
from os import path

DIR = path.dirname(path.abspath(__file__))
INSTALL_PACKAGES = open(path.join(DIR, 'requirements.txt')).read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hmdata",
    version="0.1.3",
    author="Wenjie Yin",
    author_email="yinwenjie159@hotmail.com",
    description="A package for human data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/YIN95/Human-Data-Utils',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=INSTALL_PACKAGES,
    license="MIT license",
    python_requires='>=3.5',
)

# python3 setup.py sdist bdist_wheel
# python3 -m twine upload dist/*
