#! /usr/bin/env python
#

# Copyright (C) 2011-2014 Alexandre Gramfort
# <alexandre.gramfort@telecom-paristech.fr>

import os
from os import path as op

import setuptools  # noqa; we are using a setuptools namespace
from numpy.distutils.core import setup

# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(os.path.join('qrs_detect', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = """Realtime QRS Detector and Predictor"""

DISTNAME = 'qrs_detect'
DESCRIPTION = descr
MAINTAINER = '@fraimondo'
MAINTAINER_EMAIL = 'federaimondo@gmail.com'
URL = 'http://github.com:fraimondo/qrs-detect.git'
LICENSE = 'Copyright'
DOWNLOAD_URL = 'http://github.com:fraimondo/qrs-detect.git'
VERSION = version


if __name__ == "__main__":
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=['Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'License :: OSI Approved',
                       'Programming Language :: Python',
                       'Topic :: Software Development',
                       'Topic :: Scientific/Engineering',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          platforms='any',
          packages=[
              'qrs_detect',
          ],
          # XXX put stuff here that's not gonna be shipped
          package_data={'qrs_detect': []},
          scripts=[])
