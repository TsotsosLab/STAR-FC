#!/bin/bash
# Replace $CAT2K_ROOT with the root folder of cat2k dataset
# Replace $STAR_FC_ROOT with the root folder of STAR_FC
# Replace $OUTPUT_DIR with the desired output location

#run this script to generate .ini config files for each cat2k category
# sh run_cat2k.sh <cat2k_template.ini>
#pass config_file as argument, see for example test.ini

config_file=$1

for category_dir in $CAT2K_ROOT/Stimuli/*
do
	category=$(basename $category_dir)
	echo $category_dir $category

	./STAR_FC --configFile $config_file --inputDir $CAT2K_ROOT/Stimuli/$category/ --outputDir $OUTPUT_DIR/cat2k/$category
done
