pip install numpy==1.21.6 scipy==1.7.3 cython==0.29.37 joblib==1.1.1 threadpoolctl==3.1.0
pip install setuptools==59.8.0
python setup.py build_ext --inplace
pip install -e . --no-build-isolation