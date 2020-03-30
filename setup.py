import os
import sys
import re
import codecs
from setuptools import setup

if sys.version_info < (3, 7, 1):
    raise RuntimeError("Python 3.7.1 or higher required.")


def read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(os.path.dirname(__file__), 'stats_dash', '__init__.py')
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        else:
            raise RuntimeError('Cannot find version in __init__.py')


here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='stats_dash',
    version=read_version(),
    description=('A dashboard for market data statistics and indicator-guided manual trading.'),
    long_description=long_description,
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Financial and Insurance Industry',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Scientific/Engineering',
    ],
    platforms=['any'],
    author="Ning Liu",
    url='https://github.com/0liu/stats_dash',
    license='Apache 2.0',
    packages=['stats_dash'],
    include_package_data=True,
    python_requires='>=3.7.1',
    install_requires=['pytz>=2018.9', 'pymysql>=0.9.3', 'SQLAlchemy>=1.3.1', 'pandas>=0.24.2', 'xlwings>=0.15.4', 'pika==0.13.0', 'avro-python3>=1.8.2', 'coloredlogs>=10.0'],
    keywords=('algorithmic quantitative trading finance')
)
