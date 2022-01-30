from setuptools import setup, find_packages

setup(
  name = 'parallel-pytorch',
  packages = find_packages(exclude=[]),
  include_package_data = True,
  version = '0.0.1',
  license='MIT',
  description = 'Parallel Pytorch',
  author = 'Jacob Merizian',
  author_email = 'jmerizia@gmail.com',
  url = 'https://github.com/jmerizia/parallel-pytorch',
  install_requires=[
    'mpi4py',
    'ftfy',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)