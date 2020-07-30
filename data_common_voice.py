#!/usr/bin/env python
from data import Transcript, WavFile, get_basename, ASRDataset
import pandas as pd
import glob, os
import uuid


class CommonVoiceDF(object):
    def __init__(self, path):
        self._datapath = path
        ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']
        df = pd.read_csv(self._datapath, sep='\t', header=0, names=ids)
        ASRDataset.add_uuid(df)
        ASRDataset.add_table_name(df, 'common_voice')
        df['transcript_path'] = [os.path.abspath(path)] * len(df)        
        df['duration'] = df['audio_path'].apply(lambda x: WavFile.get_wav_info(x)[1])
        self._df = df
    
    @property
    def df(self):
        return(self._df)

    
