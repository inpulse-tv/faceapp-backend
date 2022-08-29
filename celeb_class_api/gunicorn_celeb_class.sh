#!/bin/sh
export NUMBER_WORKERS=10
export NUMBER_THREADS=1
gunicorn --workers=${NUMBER_WORKERS} --threads=${NUMBER_THREADS} app:app -b 0.0.0.0:5002
