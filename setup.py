import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyBounds",                        # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Arie Beresteanu, Yunyun Lv",    # Full name of the author
    description="Implementing partial identification estimators",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    url = "https://github.com/yunyunlv/PyBounds",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    # py_modules=["PyBounds"],                # Name of the python package
    # package_dir={'':'src'},        # Directory of the source code of the package
    install_requires=['matplotlib','numpy','pandas','statsmodels'] # Install other dependencies if any
)