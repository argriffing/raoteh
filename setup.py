#!/usr/bin/env python
"""raoteh setup.py docstring"""

import distutils.core

# Return the git version as a string
def git_version():
    def _minimial_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'
    return GIT_REVISION

def setup_package():
    distutils.core.setup(
            name='raoteh',
            version='version',
            long_description='long description',
            license='license',
            author='author',
            platforms=['Windows', 'Linux', 'Solaris', 'Maac OS-X', 'Unix'],
            author_email='author email',
            keywords=['hello', 'keywords'],
            url='url',
            packages=['raoteh', 'raoteh.sampler'],
            package_data={'raoteh': ['raoteh/sampler/tests']},
            )

if __name__ == '__main__':
    setup_package()

