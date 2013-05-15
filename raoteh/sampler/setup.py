#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

from os.path import join

def configuration(parent_package='', top_path=None):
    from numpy.distutils.system_info import get_info, NotFoundError
    from numpy.distutils.misc_util import Configuration
    config = Configuration('sampler', parent_package, top_path)
    config.add_data_dir('tests')
    config.add_data_dir('benchmarks')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    #from sampler_version import sampler_version
    setup(version='1.2.3', **configuration(top_path='').todict())
