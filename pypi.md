# How to

my Pypi token:

1) Delete existing dist folder


2) Change version in pyproject to a newer version...

3) build the package

python3 -m build


4) upload the package to pypi test.

python3 -m twine upload --repository testpypi dist/*

Rt3x6RyF

5)
can now view it at, e.g., https://test.pypi.org/project/

6) 
to install:

python3 -m pip install -i https://test.pypi.org/simple/
