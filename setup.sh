#!/bin/bash
ENV_NAME="plasma"

conda env remove -n $ENV_NAME
yes | conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
pip install -r requirements.txt
mkdir bic
