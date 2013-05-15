#!/usr/bin/env python
"""raoteh setup.py docstring"""

import distutils.core

def setup_package():
    distutils.core.setup(
            name='raoteh',
            version='version',
            long_description='long description',
            license='license',
            author='author',
            platforms=['all'],
            author_email='author email',
            keywords=['hello', 'keywords'],
            url='url',
            packages=['raoteh'],
            )

if __name__ == '__main__':
    setup_package()
