#!/bin/sh
params="$@"
echo parameters are $params
python3.7 process.py /input /output $params