#!/bin/bash

set -ex

cd /tmp/ && wget https://s3.us-east-2.amazonaws.com/mxnet-scala/scala-example-ci/RNN/obama.zip && unzip obama.zip

