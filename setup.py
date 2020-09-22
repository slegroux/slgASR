from setuptools import setup
PROJECT_NAME='slgasr'
VERSION='0.0.0'
AUTHOR='Sylvain Le Groux'
EMAIL='slegroux@ccrma.stanford.edu',
setup(name=PROJECT_NAME + '-slegroux',
    version=VERSION,
    author=AUTHOR,
    # list folders, not files
    packages=['slgasr',
            'slgasr.test'],
    scripts=['slgasr/bin/run_tests.sh', 'slgasr/bin/format_commonvoice.py'],
    package_data={'slgasr': ['slgasr/data/*']},
    license='LICENSE',
    description="utils for speech recognition"
)