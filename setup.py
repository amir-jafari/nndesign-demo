import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Pedro Uria",
    author_email="pedroduriar@gmail.com",
    name='nndesign',
    license="MIT",
    description='Demos for the Neural Network Design book',
    version='v0.0.6',
    long_description=README,
    url='',
    packages=["nndesign"],
    include_package_data=True,
    python_requires=">=3.5",
    install_requires=["PyQt5==5.14.1", "numpy==1.18.1", "scipy==1.4.1", "matplotlib==3.1.2"],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries',
        'Intended Audience :: Developers',
    ],
)
