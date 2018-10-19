import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="img_ann",
    version="1.0.1",
    author="git4dev2000",
    author_email="author@example.com",
    description="Deep learning for hyperspectral image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/git4dev2000/img_ann.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)