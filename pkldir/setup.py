from setuptools import setup, find_packages

setup(
    name='pkldir',
    version='0.1.0',
    author='Center for Computation & Visualization, Brown University',
    author_email='support@ccv.brown.edu',
    packages=["pkldir", "pkldir.encoding"],
    package_data={"": ["data/two-cities.txt", "data/two-cities.pdf", "data/two-cities.docx", "data/two-cities.md"]},
    scripts=['bin/pkldir'],
    license='LICENSE.txt',
    description='A simple utility for the base64-encoding and pickling of the contents of a directory',
    long_description=open('README.md').read(),
    install_requires=[
        "python-docx",
        "textract",
    ],
)