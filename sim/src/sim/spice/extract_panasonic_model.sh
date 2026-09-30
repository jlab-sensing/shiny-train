#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path-to-zip>" >&2
    exit 1
fi

zip_path="$1"

if [[ ! -f "$zip_path" ]]; then
    echo "Error: file not found: $zip_path" >&2
    exit 1
fi

# Extract base name without directory or .zip extension
# e.g. /home/jtmadden/Downloads/EEFSX0G151E7.zip -> EEFSX0G151E7
part_name="$(basename "$zip_path" .zip)"

inner_path="${part_name}/${part_name}.lib"
out_file="${part_name}.lib"

echo "Extracting ${inner_path} -> ${out_file}"
unzip -p "$zip_path" "$inner_path" > "$out_file"
