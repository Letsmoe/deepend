import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepend",
    version="0.0.2",
    author="Moritz Utcke",
    author_email="moritz.utcke@gmx.de",
    description="Advanced end-to-end machine learning framework.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Letsmoe/deepend",
    project_urls={
        "Bug Tracker": "https://github.com/Letsmoe/deepend/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
	install_requires=["sys", "os", "numpy", "matplotlib", "cv2", "datetime", "h5py"]
)