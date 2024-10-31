#!/bin/bash

convert_to_lowercase() {
    local path="$1"
    local new_path=$(echo "$path" | tr '[:upper:]' '[:lower:]')
    
    if [ "$path" != "$new_path" ]; then
        if [ -e "$new_path" ]; then
            echo "Cannot rename '$path' to '$new_path': destination already exists"
        else
            mv "$path" "$new_path"
            echo "Renamed '$path' to '$new_path'"
        fi
    fi
    
    # Process subdirectories
    for item in "$new_path"/*; do
        if [ -d "$item" ]; then
            convert_to_lowercase "$item"
        fi
    done
}

# Start from the current directory
convert_to_lowercase "."
