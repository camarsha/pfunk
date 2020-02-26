#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='P-Funk',
      version='0.2a',
      description='Python package for Fresco UNcertainty FunKulations',
      author='Caleb Marshall',
      author_email='blah@nofuckingway.hotmail',
      url='https://github.com/dubiousbreakfast/MCMC-FRESCO',
      packages=find_packages(),
      install_requires=['pandas <= 24.0', 'matplotlib < 3.0',
                        'corner', 'emcee>2.2.0', 'seaborn', 'dynesty']
     )
