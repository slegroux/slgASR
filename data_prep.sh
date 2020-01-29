#!/usr/bin/env bash

set -x
# common voice
common_voice_dir=${1:-'/home/workfit/Sylvain/Data/Spanish/common_voice'}
pushd $common_voice_dir
    
    mkdir -p clips_16k
    mkdir -p transcripts_bu
    for i in *.tsv; do
        cp $i transcripts_bu/$i.og
    done
    abs_path=$(realpath clips_16k)
    for i in *.tsv; do
        #sed -i 's/.mp3/.wav/g' $i
        sed -i "s|\(common_voice_es_[0-9]*\).mp3|${abs_path}/\1.wav|g" $i
    done
    
    for i in clips/*.mp3; do
        sox $i -t wav -r 16k -b 16 -e signed clips_16k/$(basename $i .mp3).wav
    done
popd
