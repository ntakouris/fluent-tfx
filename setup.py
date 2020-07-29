import setuptools

setuptools.setup(
    name="fluent-tfx",
    version="0.0.1",
    author="Theodoros Ntakouris",
    author_email="zarkopafilis@gmail.com",
    description="A fluent API layer for tensorflow extended e2e machine learning pipelines",
    long_description_content_type="text/markdown",
    url="https://github.com/ntakouris/fluent-tfx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
