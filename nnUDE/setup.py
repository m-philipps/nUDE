import os
import re
import sys

from setuptools import find_packages, setup


def read(fname):
    """Read a file."""
    return open(fname).read()


# read version from file
__version__ = ''
version_file = os.path.join('nnUDE', 'version.py')
# sets __version__
exec(read(version_file))  # pylint: disable=W0122 # nosec

# project metadata
# noinspection PyUnresolvedReferences
setup(
    name='nnUDE',
    version=__version__,
    description='nnUDE: scripts to experiment with non-negative universal differential equations.',
    long_description_content_type="text/markdown",
    author='Dilan Pathirana',
    author_email='dilan.pathirana@uni-bonn.de',
    packages=find_packages(exclude=['doc*', 'test*']),
    install_requires=[
        'more-itertools',
        'numpy',
        'pandas',
        'pyyaml',
        'python-libsbml>=5.17.0',

        'amici',
        'fides',
        'pypesto',
        'petab',

        'notebook',
    ],
    include_package_data=True,
    python_requires='>=3.11.1',
    extras_require={
        'test': [
            'pytest >= 5.4.3',
        ],
    },
)
