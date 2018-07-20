How to Build Documentation
==========================

First, make sure this is your current working directory::

   cd doc/

Then follow the instructions for your operating system. To view the
documentation after building, open ``_build/html/index.html`` in your web
browser.


Debian/Ubuntu
-------------

To make the html documentation, install the prerequisites::

    apt-get install python-sphinx texlive-latex-recommended dvipng librsvg2-bin imagemagick docbook2x

and do::

    make html

and to view it, do::

    epiphany _build/html/index.html

Fedora
------

Fedora (and maybe other RPM based distributions), install the prerequisites::

    dnf install python3-sphinx librsvg2 ImageMagick docbook2X texlive-dvipng-bin texlive-scheme-medium librsvg2-tools 

After that, run::

    make html

If you get **mpmath** error, install python3-mpmath package::

    dnf install python3-mpmath

And view it at::

    _build/html/index.html


OSX/Mac
-------

First, install Python on your system if you have not already. A graphical
installer is available on the official `Python website`_. Alternatively you may
want to install the `Anaconda`_ or `Miniconda`_ distributions.

.. _Python website: https://www.python.org/downloads/mac-osx/
.. _Anaconda: https://www.anaconda.com/download/#macos
.. _Miniconda: https://conda.io/miniconda.html

Then install the Sphinx package through pip:

   pip install sphinx

Additional non-Python prerequisites on OSX can be installed through `Homebrew`_.
Install the Homebrew tool using the instructions on the web site, then run the
following commands::

   brew install imagemagick librsvg

.. _Homebrew:: https://brew.sh/
