import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="fluent-tfx",
    version="0.23.1",
    author="Theodoros Ntakouris",
    author_email="zarkopafilis@gmail.com",
    description="A fluent API layer for tensorflow extended e2e machine learning pipelines",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/ntakouris/fluent-tfx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
