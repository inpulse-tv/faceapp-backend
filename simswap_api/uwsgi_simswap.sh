#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate simswap
export NUMBER_PROCESSES=10
export NUMBER_THREADS=1
uwsgi --socket 0.0.0.0:5001 --processes=${NUMBER_PROCESSES} --threads=${NUMBER_THREADS} --protocol=http -w  api_swap:app
