from setuptools import setup, find_packages

setup(
    name="pose-utils",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.17.3",
        "transformations==2019.4.22",
        "matplotlib==3.1.1",
        "more-itertools==7.2.0",
    ],
)
