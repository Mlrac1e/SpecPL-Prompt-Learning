#!/bin/bash
# Optional placeholder for VAE-cache preprocessing.
# The public training scripts build the SpecPL VAE teacher cache on demand.

DATA_ROOT=${DATA_ROOT:-path/to/data}
OUTPUT_ROOT=${VAE_CACHE_ROOT:-path/to/output_vae_data}

echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "No standalone VAE preprocessing script is required for the maintained release path."