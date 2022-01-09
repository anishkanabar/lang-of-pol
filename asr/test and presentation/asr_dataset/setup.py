import setuptools

setuptools.setup(
    name='asr-dataset',
    version='0.1',
    description='Library to serve datasets to asr modules.',
    url='#',
    author='Eric Chandler',
    author_email='echandler@uchicago.edu',
    install_requires=['pandas','librosa','soundfile'],
    packages=setuptools.find_packages(),
    zip_safe=False)
