API Reference
=============

nndesigndemos is a GUI application rather than a library, so its public API is
intentionally tiny: one function starts the main menu, and everything else is
reached by clicking through the windows it opens.

Entry point
-----------

.. autofunction:: nndesigndemos.nndtoc

   Start the Qt application and show the main menu.

   This is the only function you normally need. It creates the
   ``QApplication``, sizes the windows from the primary screen, opens the
   book-chooser window and then blocks until the last window is closed.
   It calls :func:`sys.exit` when the application quits, so run it from a
   script or an interactive shell rather than from code that must continue
   afterwards.

   :param bool play_sound: Play the sound effects used by some demos.
       Pass ``False`` to mute them.

Main windows
------------

These classes are created by :func:`nndesigndemos.nndtoc` and are documented
here for people who want to embed or script the menu. They are not part of the
stable public interface.

.. autoclass:: nndesigndemos.nndesign.MainWindow

.. autoclass:: nndesigndemos.Window.MainWindowNN

.. autoclass:: nndesigndemos.Window.MainWindowDL
