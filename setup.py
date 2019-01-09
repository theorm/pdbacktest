from setuptools import setup

setup(name='pdbacktest',
      version='0.1',
      description='Quick and dirty backtesting function for pandas/numpy using backtrader',
      url='http://github.com/theorm/pdbacktest',
      author='Roman Kalyakin',
      author_email='roman@kalyakin.com',
      license='MIT',
      packages=['pdbacktest'],
      install_requires=[
        'backtrader'
      ])
