from setuptools import setup
PROJECT_NAME='slgasr'
VERSION='0.0.1'
AUTHOR='Sylvain Le Groux'
EMAIL='slegroux@ccrma.stanford.edu'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name=PROJECT_NAME + '-slegroux',
    version=VERSION,
    author=AUTHOR,
    # list folders, not files
    packages=['slgasr',
            'slgasr.test'],
    scripts=['slgasr/bin/run_tests.sh', 'slgasr/bin/format_commonvoice.py'],
    package_data={'slgasr': ['slgasr/data/*']},
    license='LICENSE',
    description="A set of utilities for speech recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slegroux/slgASR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)