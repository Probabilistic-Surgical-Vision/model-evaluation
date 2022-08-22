#!/bin/bash

find . -not \( \
    -path "./.git*" \
    -or -path "./.gradient*" \
    -or -path "./models*" \
    -or -name "." \
    -or -name ".." \) | xargs rm -rf
