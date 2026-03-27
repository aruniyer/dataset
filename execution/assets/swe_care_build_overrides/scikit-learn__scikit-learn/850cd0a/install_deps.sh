pip install numpy==1.15.4 scipy==1.1.0 cython==0.29.37 joblib==1.1.1
python setup.py build_ext --inplace
pip install -e . --no-build-isolation