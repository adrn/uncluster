from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='uncluster',
      version='0.1.dev',
      author='adrn',
      author_email='adrn@princeton.edu',
      url='https://github.com/adrn/uncluster',
      license="License :: OSI Approved :: MIT License",
      description='Disrupting globular clusters.',
      long_description=long_description,
      packages=['uncluster'],
)
