import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
  name='gradoptics',
  version='0.0.4',
  description='End-to-end differentiable optics',
  long_description=long_description,
  long_description_content_type="text/markdown",
  author='Sean Gasiorowski, Michael Kagan, Maxime Vandegar, Sanha Cheong',
  author_email='sgaz@slac.stanford.edu, makagan@slac.stanford.edu, maxime.vandegar@slac.stanford.edu, sanha@stanford.edu',
  url="https://github.com/magis-slac/gradoptics",
  classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
  package_dir={'': 'src'},
  packages=setuptools.find_packages(where="src"),
)
