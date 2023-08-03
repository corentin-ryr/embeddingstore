import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="embeddingstore",
    version="0.1.1",
    author="C. Royer",
    author_email="",
    description="A fork of the embedding store python package by Microsoft",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/corentin-ryr/embeddingstore",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "langchain",
        "openai",
        "azure-identity",
        "azure-keyvault",
        "azure-storage-blob",
        "azure-core",
        "azure-ai-ml",
        "opencensus-ext-azure",
        # Add other dependencies as needed
    ],
)
