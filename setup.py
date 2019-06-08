from setuptools import setup


setup(name='LEAN',
      version='0.0.1',
      url='https://github.com/coetaur0/ESIM',
      license='Apache 2',
      author='Aurelien Coet',
      author_email='aurelien.coet19@gmail.com',
      description='Implementation of the LEAN model for NLI with PyTorch',
      packages=[
        'lean'
      ],
      install_requires=[
        'wget',
        'numpy',
        'nltk',
        'matplotlib',
        'tqdm',
        'torch'
      ])
