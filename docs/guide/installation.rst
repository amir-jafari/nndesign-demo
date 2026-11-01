Installation
============

nndesigndemos is supported on **macOS**, **Linux** and **Windows**. It requires
Python 3.9 or newer.

.. note::

   The demos are built on `PyQt6 <https://pypi.org/project/PyQt6/>`_, so your
   operating system version needs to be compatible with it. If you get an
   installation error, an unsupported Qt platform is the most likely reason.

Installing via pip
------------------

The quick way is to install straight from PyPI, which works in most cases:

.. code-block:: bash

   pip install nndesigndemos

The recommended way is to create a virtual environment first, so the pinned
dependencies cannot conflict with other packages on your system.

**macOS / Linux**

.. code-block:: bash

   python3 -m venv env
   source env/bin/activate
   pip install nndesigndemos

**Windows**

.. code-block:: bat

   python -m venv env
   env\Scripts\activate.bat
   pip install nndesigndemos

To leave the virtual environment later, just run ``deactivate``.

Dependencies
------------

``pip`` installs these packages automatically. The versions listed are known to
work; older or newer versions will usually work too.

- Python 3.9+
- PyQt6 6.7.1
- NumPy 2.0.1
- SciPy 1.13.1
- Matplotlib 3.9.2

Platform notes
--------------

Linux: missing Qt ``xcb`` plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   On Linux, installing or launching the demos may fail with an error similar
   to the following:

   .. code-block:: text

      qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
      This application failed to start because no Qt platform plugin could be initialized.
      Reinstalling the application may fix this problem.

      Available platform plugins are: eglfs, offscreen, wayland, wayland-egl, linuxfb, minimal, xcb, minimalegl, vkkhrdisplay, vnc.

      Aborted (core dumped)

   Install the missing system library and try again:

   .. code-block:: bash

      sudo apt-get install -y libxcb-cursor-dev

Sound support
^^^^^^^^^^^^^

Some demos play short sound effects. On macOS or Linux you may additionally
need an audio backend such as PulseAudio:

.. code-block:: bash

   sudo apt-get install pulseaudio

If you would rather run without sound, launch the menu with
``nndtoc(play_sound=False)`` as described in :doc:`quickstart`.

Verifying the installation
--------------------------

Open a Python shell inside the environment where you installed the package and
start the main menu:

.. code-block:: python

   from nndesigndemos import nndtoc
   nndtoc()

A window titled *Neural Network Design & Deep Learning Demos* should appear
with a button for each book. Close the window to return to the shell.

Getting help
------------

If something does not work, please open an issue on the
`GitHub issue tracker <https://github.com/amir-jafari/nndesign-demo/issues>`_.
Include your operating system, Python version and the full error message.
