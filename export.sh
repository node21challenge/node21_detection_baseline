#!/usr/bin/env bash

. ./build.sh

docker save noduledetector | gzip -c > noduledetection.tar.gz
