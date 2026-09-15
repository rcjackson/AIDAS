.. highlight:: shell

============
Installation
============


Stable release
--------------

To install Argonne AI-Driven Adaptive Sampling System, run this command in your terminal:

.. code-block:: console

    $ pip install aidas-atmos

This is the preferred method to install Argonne AI-Driven Adaptive Sampling System, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for Argonne AI-Driven Adaptive Sampling System can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone https://github.com/rcjackson/AIDAS.git

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/rcjackson/AIDAS/tarball/main

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ pip install .


.. _Github repo: https://github.com/rcjackson/AIDAS
.. _tarball: https://github.com/rcjackson/AIDAS/tarball/main


Migrating from adam-atmos
-------------------------

Through version 0.5.0 this project was released on PyPI as ``adam-atmos``, under
the name ATMOS Analogue Digital Twin (ADAM), and imported as ``adam``. From
version 0.6.0 it is released as ``aidas-atmos`` and imported as ``aidas``.

PyPI distribution names cannot be renamed in place, so ``adam-atmos`` remains a
separate project. Its final release, 0.5.1, contains no functionality of its
own: it depends on ``aidas-atmos`` and forwards ``import adam`` to ``aidas``
with a ``DeprecationWarning``. Existing code will keep working, but the shim
will be removed in a future release.

To migrate:

.. code-block:: console

    $ pip uninstall adam-atmos
    $ pip install aidas-atmos

and update your imports:

.. code-block:: python

    import aidas  # was: import adam

Every module, class and function keeps its name; only the top-level package
changed, so ``adam.io.preprocess_radar_image`` becomes
``aidas.io.preprocess_radar_image``, and so on.

Releases of ``adam-atmos`` up to and including 0.5.0 remain installable from
PyPI for reproducibility, but will not receive further updates.
