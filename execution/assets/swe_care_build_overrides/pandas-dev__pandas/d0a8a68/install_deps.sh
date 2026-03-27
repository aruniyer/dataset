export CFLAGS="${CFLAGS} -Wno-error=array-bounds"
pip install 'Cython>=0.29.16,<3'
pip install 'python-dateutil>=2.7.3' 'pytz>=2017.3' 'numpy==1.17.3'
python setup.py build_ext --inplace -j 4
python setup.py develop