#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:09:48 2024

@author: mohamadjouni
"""

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='FAnomAlly_utils',
    version='0.0.1',
    description='Utility package for detecting anomalies',
    long_description=readme,
    author='Emmanuel Gangler, Mohamad Jouni',
    author_email='mohamad.jouni@etu.umontpellier.fr',
    url='https://github.com/ManuGangler/FAnomAlly/FAnomAly_utils',
    license=license,
    packages=find_packages()
)