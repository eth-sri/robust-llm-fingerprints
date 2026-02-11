# Copyright (c) 2025 Thibaud Gloaguen and contributors
# Licensed under the Responsible AI SOURCE CODE License, Version 1.1
# (see LICENSE_CODE).

from setuptools import setup, find_packages

# Install the project using the standard ``src`` layout so the public package
# becomes ``robust_fp`` instead of the previous ``src`` namespace.
setup(
    name="domain-watermark",
    version="0.1.0",
    license="SEE LICENSE IN LICENSE_CODE",
    license_files=["LICENSE_CODE"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
