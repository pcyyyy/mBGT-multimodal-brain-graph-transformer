#!/usr/bin/env bash



# install requirements
pip install torch
pip install lmdb
pip install torch-scatter==2.0.9
pip install torch-sparse==0.6.12
pip install torch-geometric==1.7.2


cd fairseq
pip install . --use-feature=in-tree-build
python setup.py build_ext --inplace
