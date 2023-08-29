#!/bin/bash

cat forecastdemand_nsw.csv.zip* > ./forecastdemand_nsw.csv.zip

unzip forecastdemand_nsw.csv.zip

unzip temperature_nsw.csv.zip

unzip totaldemand_nsw.csv.zip

rm forecastdemand_nsw.csv.zip
rm -rf __MACOSX