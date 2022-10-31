#!/bin/sh

echo "Adding current repo to PYTHONPATH"
echo "==="
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo $PYTHONPATH
echo "==="

echo "Start Training"
echo "==="
cd src
python train.py