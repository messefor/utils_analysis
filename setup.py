
import os
import glob

from setuptools import setup, find_packages


def _requirements_from_text(path):
  with open(path, 'r') as f:
    return f.read().splitlines()


NAME = 'Odutils'
VERSION = '0.1.0'
URL = 'https://github.com/messefor/utils_analysis.git'
DESC = 'My utils for analysis.'

LICENSE = 'MIT'
AUTHOR = 'Daisuke Oda'
AUTHOR_EMAIL = 'messefor@gmail.com'


setup(

  name=NAME,
  version=VERSION,
  license=LICENSE,
  description=DESC,
  url = URL,

  author=AUTHOR,
  author_email=AUTHOR_EMAIL,

  packages=find_packages('src'),
  package_dir={'': 'src'},
  install_requires=_requirements_from_text('requirements.txt'),
  setup_requires=['pytest-runner'],
  tests_require=['pytest'],
  test_suite='tests'

)