Quickstart
==========

Launching the main menu
-----------------------

All demos start from the same main menu. Open a Python shell and run:

.. code-block:: python

   from nndesigndemos import nndtoc
   nndtoc()

A window will pop up where you choose a book, then a chapter, then a demo.

.. figure:: /_static/screenshots/main_menu.webp
   :alt: The nndesigndemos main menu
   :align: center
   :width: 60%

   The main menu, with one button per book.

Running without sound
---------------------

Several demos play sound effects. To mute them, pass ``play_sound=False``:

.. code-block:: python

   from nndesigndemos import nndtoc
   nndtoc(play_sound=False)

How the menu is organized
-------------------------

The menu follows the structure of the books:

1. **Book.** The first window offers two buttons, one for *Neural Network
   Design* and one for *Neural Network Design: Deep Learning*.
2. **Chapter.** Each book opens a table of contents that shows four chapters at
   a time. Buttons along the bottom (``2-5``, ``6-9``, ``10-13``, ...) switch
   between blocks of chapters.
3. **Demo.** Every chapter has a drop-down list of its demos. Selecting an
   entry opens that demo in its own window. You can keep several demos open
   at the same time.

.. figure:: /_static/screenshots/toc_book1.webp
   :alt: The Neural Network Design table of contents window
   :align: center
   :width: 60%

   The table of contents for *Neural Network Design*, showing chapters 2 to 5.

For every section of the *Neural Network Design* book where you see the MATLAB
logo, there is a corresponding Python demo in this package. The demos for the
second book are still in progress.

.. tip::

   Each demo window has a short explanation of what it shows and which book
   section it belongs to, so you can read the relevant pages alongside it.

Multiple monitors
-----------------

.. note::

   If you are using multiple monitors and switching between them, you may need
   to restart your computer to avoid scaling issues. The windows size
   themselves from the primary screen when the menu starts.

Next steps
----------

- Browse the :doc:`tutorials </tutorials/index>` for chapter-by-chapter walkthroughs.
- See the :doc:`API reference </api/index>` if you want to launch the menu from your own scripts.
