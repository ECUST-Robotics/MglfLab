from setuptools import find_packages, setup


setup(
    name="mglf-lab",
    version="0.1.0",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    python_requires=">=3.11",
)
