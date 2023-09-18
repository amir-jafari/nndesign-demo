## Steps of Updating Project Version in PyPI

#### Date: Sep 18, 2023

Here is what I did to push the repo to the PyPI. I followed these two instructions:


* https://packaging.python.org/en/latest/tutorials/packaging-projects/

* https://towardsdatascience.com/publishing-your-own-python-package-3762f0d268ec


Since the project is already in PyPI, we only need to update it to a new version. So I skipped some initial steps. Here is what I did: 

1. Update `setup.py` and `README.md`. 
   * I updated the `author`, `version`, `python_requires`, and `classifiers` parameters in `setuptools.setup` function of `setup.py`. 
   * I also updated the Dependencies version in the `README.md` file.

2. Make sure you install the latest versions of `setuptools` and `wheel`:

    * ```python3 -m pip install --user --upgrade setuptools wheel```

3. Build the distribution files:
   * `python3 setup.py sdist bdist_wheel`

4. Create a text file named `.pypirc` with the following content. Don't forget to change your own username and password. And I didn't set testpypi.

```
[distutils]
index-servers =
  pypi
[pypi]
username: your_pypi_username
password: your_pypi_password
```

5. Install twine and upload the package to PyPI.
    * `python3 -m pip install --upgrade twine`
    * `python3 -m twine upload dist/*`

6. After uploading, use pip install to test it by installing it in a new virtual environment