from setuptools import setup

setup(
    name = "irtk",
    packages=["irtk"],
    version = "1.0",
    #data_files = [ ('irtk', ['build/_irtk.so']) ],
    #package_data = { 'irtk' : ['build/_irtk.so'] },
    package_data = { 'irtk' : ['irtk/_irtk.so'] },
    include_package_data=True,
    author = 'Kevin Keraudren',
    description = 'Cython based wrapper for IRTK',
    license = 'GPLv2',
    keywords = 'IRTK cython',
    url = 'https://github.com/BioMedIA/python-irtk'
    )
