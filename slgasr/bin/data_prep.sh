#!/usr/bin/env bash

set -x
# common voice
common_voice_dir=${1:-'/home/workfit/Sylvain/Data/Spanish/common_voice'}
kaldi_data_dir=$KALDI_SLG/egs/commonvoice_spanish/s5/data

pushd $common_voice_dir    
 
    if [ ! -d transcripts_bu ]; then
        echo "# Archiving original transcripts"
        mkdir -p transcripts_bu
        for i in *.tsv; do
            cp $i transcripts_bu/$i.og
        done
    fi

    abs_path=$(realpath clips_16k)
    for i in *.tsv; do
        #sed -i 's/.mp3/.wav/g' $i
        sed -i "s|\(common_voice_es_[0-9]*\).mp3|${abs_path}/\1.wav|g" $i
    done
    
    if [ ! -d clips_16k ]; then
        echo "# convert mp3 to 16k wav"
        mkdir -p clips_16k
        for i in clips/*.mp3; do
            sox $i -t wav -r 16k -b 16 -e signed clips_16k/$(basename $i .mp3).wav
        done
    fi
popd

for dataset in {train,test}; do
    if [ -d ${kaldi_data_dir}/${dataset} ]; then
        mkdir -p ${kaldi_data_dir}/${dataset} 
    fi
python - <<EOF
from data import ASRDataset
ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']
asr_data = ASRDataset.init_with_csv('${common_voice_dir}/${dataset}.tsv', ids, name='common_voice', lang='spanish')
asr_data.export2kaldi('${kaldi_data_dir}/${dataset}')
EOF
    sed -i 's/\"//g' ${kaldi_data_dir}/${dataset}/text
done

pushd $KALDI_SLG/egs/commonvoice_spanish/s5
    . ./cmd.sh
    . ./path.sh
    for dataset in {train,set}; do
        utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk | sort >data/${dataset}/spk2utt
        utils/fix_data_dir.sh data/${dataset}
    done
popd