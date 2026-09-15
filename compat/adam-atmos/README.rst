====================================
adam-atmos is now aidas-atmos
====================================

``adam-atmos`` has been renamed to ``aidas-atmos``, the Argonne AI-Driven
Adaptive Sampling System (AIDAS). The import package ``adam`` is now ``aidas``.

This release of ``adam-atmos`` contains no functionality of its own. It depends
on ``aidas-atmos`` and forwards ``import adam`` to ``aidas`` with a
``DeprecationWarning``, so existing code keeps working while you migrate.

Please switch to::

    pip install aidas-atmos

and::

    import aidas

* Source code: https://github.com/rcjackson/AIDAS
* Documentation: https://rcjackson.github.io/AIDAS/
