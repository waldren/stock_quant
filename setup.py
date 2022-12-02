# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='stock_quant',
    version='0.1.0',
    description='Project to house code to analyze stocks',
    long_description=readme,
    author='Steven E. Waldren',
    author_email='swaldren@gmail.com',
    url='https://github.com/swaldren',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

