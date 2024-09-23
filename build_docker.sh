#!/bin/bash
# Build Docker image

if [ $# -ne 3 ]; then
  echo "ERROR USAGE: $0 <build-directory-relative-path> <dockerhub username> <cache:'' or '--no-cache'>"
  echo You need to specify a build directory and a dockerhub username
  exit 1
fi

echo working directory is $(pwd)
echo directory to build is $1

docker build ${3} -t ${2}/gradoptics:latest $1