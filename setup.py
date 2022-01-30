from setuptools import setup, find_packages

setup(
  name = 'parallel-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.2',
  license='MIT',
  description = 'utilities for working with MPI+PyTorch',
  author = 'Jacob Merizian',
  url = 'https://github.com/jmerizia/parallel-pytorch',
  install_requires=[
    'mpi4py',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)
