#!/bin/bash

# A script for creating utt2path and utt2label files, required by the lidbox tool.

if [ $# != 1 ]; then
	echo "Usage: ./generate_utts.sh [WAV_DIR]"
	echo ""
	echo "Creates utt2path and utt2label files from the files"
	echo "in the WAV_DIR. All files within WAV_DIR are assumed"
	echo "to be .wav files. The 'utt' ids are formed from the"
	echo "basenames of the wav files. In absence of true labels,"
	echo "all labels are set to 'en' because lidbox requires"
	echo "placeholder labels."
	exit 1;
fi

SRCDIR="$1"

echo "Creating utt2path and utt2label from '$SRCDIR'."
if [ ! -f utt2label ] && [ ! -f utt2path ]; then
	echo "Writing new utt2label and utt2path files."
	for i in $(find "$PWD/$SRCDIR" -type f)
	do
		FNAME=$(basename $i)
		echo "${FNAME%.*} ${i}" >> utt2path
		echo "${FNAME%.*} en" >> utt2label	
	done
	echo "Finished."
else
	echo "An utt2* file exists already, will not overwrite. Remove both utt2* files and try again."
fi

