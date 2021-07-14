#!/bin/bash

find fixed_notebooks/ -type f ! -name "*.py" | xargs rm &> /dev/null

rm -rf exec-logs/
rm -rf fixed_notebooks/
rm -rf patches/
rm relancer_fqns.json
