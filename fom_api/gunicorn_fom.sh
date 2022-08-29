#!/bin/sh
export NUMBER_WORKERS=10
export NUMBER_THREADS=1
export TIMEOUT=120
gunicorn --workers=${NUMBER_WORKERS} --threads=${NUMBER_THREADS} --bind 0.0.0.0:5000 --timeout ${TIMEOUT}  fom_api:app