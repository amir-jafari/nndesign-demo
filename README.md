# nndesigndemos

This is a set of demonstrations paired with the [Neural Network Design](https://hagan.okstate.edu/nnd.html) 
& Neural Network Design: Deep Learning books written in Python.

## Installation

nndesigndemos is supported on macOS, Linux and Windows. It uses PyQt5, so your OS version needs to be compatible with it.
 If you get an installation error, this is most likely the reason.

### Installing via pip

The quick way is simply to install via `pip install nndesigndemos`, which works in most cases.

The recommended way is to create a virtual environment to avoid dependency issues. Here is an easy way to do so:

```
python3 -m venv env
source env/bin/activate  # macOS/Linux
env\Scripts\activate.bat  # Windows
pip install nndesigndemos
```

To deactivate the virtual environment, just type `deactivate`.

## Usage

All the demos start from the same main menu, which can be accessed by entering the Python Shell and running

```
from nndesigndemos import nndtoc
nndtoc()
```

After doing so, a window will pop up, and you will be able to navigate the demos listed by book and then by chapter.

There are some demos that have sound, so if you want to mute them just run `nndtoc(play_sound=False)` instead.

The original software for these demos runs on MATLAB, so for every section of the 
[Neural Network Design](https://hagan.okstate.edu/NNDesign.pdf) book where you see the MATLAB logo, 
there will be a corresponding Python demo in this package. The second book is in progress.

If you are using multiple monitors and switching between them, you may need to restart your computer to avoid scaling issues.

## Dependencies

These are the packages needed to run all the demos. These specific versions are known to work, but this does not mean 
older or newer versions will cause any issues.

- Python 3.5+
- PyQt5 5.14.1
- NumPy 1.18.1
- SciPy 1.4.1
- Matplotlib 3.1.2

## License

nndesigndemos is available under MIT license.
