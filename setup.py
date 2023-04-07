#!/usr/bin/env python3

import os
from setuptools import setup



# https://setuptools.pypa.io/en/latest/index.html
setup(
    name='CEWS',
    description='Cut Weight Statistic Calculator',
    version='0.0.1',
    author='Lesley Wheat',
    author_email='wheatd@mcmaster.ca',
    url='https://github.com/LesleyWheat/CAS741_CEWS',
    packages=['CEWS'],
    include_package_data=True,
    python_requires=">=3.9.*",
    install_requires=['numpy', 'scipy'],
    license='-',
    zip_safe=False,
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.9',
    ]
)