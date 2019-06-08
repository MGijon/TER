from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
     name='WER',
     version='0.1',
     scripts=['principal',
              'graphics'] ,
     author="Manuel Gijón Agudo",
     author_email="manuel.gijon@outlook.es",
     description="bla bla bla bla",
     long_description="bla bla bla bla bla bla bla bla",
     long_description_content_type="text/markdown",
     url="https://github.com/MGijon/TER",
     packages=find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3.6.5",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     install_requires=[
          'markdown',
          'matplotlib.pyplot',
          'logging',
          'random',
          'sklearn.metrics',
          'pandas',
          'nltk.corpus',
          'os',
      ],
 )


# https://dzone.com/articles/executable-package-pip-install
# https://python-packaging.readthedocs.io/en/latest/dependencies.html
