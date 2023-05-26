import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="one_peace",
    py_modules=["one_peace"],
    version="1.0",
    description="",
    author="M6",
    packages=find_packages(exclude=["one_peace_vision"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
