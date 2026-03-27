pip install 'Cython<3'
pip install 'python-dateutil>=2' 'pytz>=2011k' 'numpy==1.12.1'
python setup.py build_ext --inplace -j 4
python setup.py develop