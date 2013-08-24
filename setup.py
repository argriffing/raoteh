#!/usr/bin/env python
"""raoteh setup.py docstring"""

import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
            assume_default_configuration=True,
            delegate_options_to_subpackages=True,
            quiet=True)
    config.add_subpackage('raoteh')
    #config.add_data_files('raoteh', '*.txt')
    #config.get_version('raoteh/version.py')
    return config

def setup_package():
    #write_version_py()

    cmdclass = {}

    metadata = dict(
            name = 'raoteh',
            maintainer = 'raoteh developers',
            description = 'description',
            long_description = 'long description',
            url = 'url',
            download_url = 'download url',
            license = 'license',
            cmdclass = cmdclass,
            platforms = ['Windows', 'Linux', 'Solaris', 'Maac OS-X', 'Unix'],
            #test_suite = 'nose.collector',
            ##author='author',
            ##author_email='author email',
            ##keywords=['hello', 'keywords'],
            ##packages=['raoteh', 'raoteh.sampler'],
            ##package_data={
                ##'raoteh': ['raoteh/sampler/tests/test_*.py'],
                ##},
            )

    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                'clean')):
        # do things with few dependencies
        print 'doing things with few dependencies...'
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        #FULLVERSION, GIT_REVISION = get_version_info()
        #metadata['version'] = FULLVERSION
    else:
        print 'doing things with more dependencies...'
        from numpy.distutils.core import setup
        #cwd = os.path.abspath(os.path.dirname(__file__))
        metadata['configuration'] = configuration
    setup(**metadata)

if __name__ == '__main__':
    setup_package()

