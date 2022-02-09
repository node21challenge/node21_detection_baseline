#!/usr/bin/env bash

./build.sh

docker save node21_detection_baseline | gzip -c > node21_detection_baseline.tar.gz
