pip install 'Cython>=0.29.21,<3'
pip install 'setuptools==38.6.0'
pip install 'python-dateutil>=2.7.3' 'pytz>=2017.2' 'numpy==1.19.3'
python setup.py build_ext --inplace -j 4
python setup.py develop