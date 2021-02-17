#!/bin/bash

# Script for splitting wav file to given segments. 

if [ $# != 3 ]; then
	echo "Usage: ./split_wav_to_segments.sh [WAV_FILE] [DIAR_JSON] [OUTPUT_DIR]"
	echo ""
	echo "Splits WAV_FILE to segments defined in DIAR_JSON. Segments are saved"
	echo "in OUTPUT_DIR."
	exit 1;
fi

WAV="$1"
JSON="$2"
OUTDIR="$3"
IFS=" "
mkdir -p $OUTDIR

# Input json is assumed to be in format:
# {'results': [['segment_1_name', '5.941', '11.650'], ['segment_2_name', '12.495', '14.800'], ...]}
#
# Essentially results are a list of segments that contain segment_name/id
# (can be non-unique), segment start seconds and segment end seconds.
# Start and end seconds are measured from the beginning of the WAV_FILE.
#
# The json input transformed into a temp file with the following format:
# segment_name start_sec end_sec

#SEGMENT_TMP="segments.tmp"
#sed 's/{"results": \[\["//' $JSON > $SEGMENT_TMP
#sed -i 's/"\], \["/\n/g' $SEGMENT_TMP
#sed -i 's/", "/ /g' $SEGMENT_TMP
#sed -i 's/"\]\]}//' $SEGMENT_TMP

# Print each command from here on for debugging
set -o xtrace 
#jq -r  '.[][] | [ .[0], .[1], .[2]|tostring] | join(" ")' $JSON | while read i
#jq -r  '.[] | [ .id, .start, .end|tostring] | join(" ")' $JSON | while read i
jq -r '.[]| [ .id, (.start / .samplerate_num), (.end/.samplerate_num)|tostring ] | join(" ")' $JSON | while read i
do
	read -a arg <<< "$i"
	START=$(perl -e "printf '%08d',  ${arg[1]}*1000") # Pad with zeroes for sorting
	END=$(perl -e "printf '%08d',  ${arg[2]}*1000")
	sox $WAV -c 1 "${OUTDIR}/split_${START}_${END}_${arg[0]}.wav" trim "${arg[1]}" "=${arg[2]}"
done # < $SEGMENT_TMP

#rm $SEGMENT_TMP
