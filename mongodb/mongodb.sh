#!/bin/sh
mongoimport --db=ingeniance3 --collection=celebrities --type=json --jsonArray  --file=./volume/celebrities.json
mongoimport --db=ingeniance3 --collection=Images --type=json --jsonArray  --file=./volume/Images.json