from setuptools import find_packages, setup


setup(
    name="mglf-lab",
    version="0.1.0",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    package_data={
        "mglf_lab": [
            "data/Robots/unitree/go2w_description/urdf/*",
            "data/Robots/unitree/go2w_description/meshes/*",
            "data/LICENSES/*",
        ]
    },
    include_package_data=True,
    python_requires=">=3.11",
)
