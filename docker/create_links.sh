#!/bin/bash

# Find all files matching the pattern recursively
find /usr/lib/python3/dist-packages -type f -name '*.cpython-38-x86_64-linux-gnu.so' -print0 | while read -d '' -r file; do
    # Extract the name without extension
    name=$(basename "$file" .cpython-38-x86_64-linux-gnu.so)
    # Get the directory of the original file
    original_dir=$(dirname "$file")
    # Check if the destination file already exists
    if [ ! -L "$original_dir/$name.so" ]; then
        # Create a symbolic link
        ln -s "$file" "$original_dir/$name.so"
    fi
done