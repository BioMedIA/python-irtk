python-irtk
===========

Install
-------

The easiest way to install `python-irtk` is through `conda`:

    conda install -c kevin-keraudren python-irtk
    
`conda` is the package manager that comes with the free Anaconda Python distribution from [Continuum Analytics](http://continuum.io/downloads). It is highly recommended for scientific computing in Python as it allows you to install within seconds your favourite libraries, along with their depencies. Additionally, you can easily set up different virtual environments in order to test different versions of some libraries.

Documentation
-------------

See the [documentation](http://biomedia.github.io/python-irtk/)
and [notebooks](http://nbviewer.ipython.org/github/BioMedIA/python-irtk/tree/master/doc/notebooks/).

To update the documentation:

    cd python-irtk/doc/
    make html
    cd ../../python-irtk-doc/
    git add .
    git commit -m "rebuilt docs"
    git push origin gh-pages

See [daler's](http://daler.github.io/sphinxdoc-test/includeme.html) tutorial for
more details on publishing sphinx-generated docs on github.

For more information on IRTK:
-----------------------------

See the github repository of [IRTK](https://github.com/BioMedIA/IRTK).

