#!/bin/bash
mkdir codeql-home

wget https://github.com/github/codeql-cli-binaries/releases/download/v2.20.6/codeql.zip -P codeql-home/
unzip codeql-home/codeql.zip -d codeql-home/

git clone https://github.com/github/codeql.git codeql-home/codeql-repo
cd codeql-home/codeql-repo

echo 'export PATH="$(pwd)/codeql-home/codeql:$PATH"' >> ~/.bashrc 
source ~/.bashrc

codeql resolve languages
codeql resolve qlpacks