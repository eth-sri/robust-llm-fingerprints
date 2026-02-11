from setuptools import setup, find_packages

# Install the project using the standard ``src`` layout so the public package
# becomes ``robust_fp`` instead of the previous ``src`` namespace.
setup(
    name="domain-watermark",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
