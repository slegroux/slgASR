#!/usr/bin/env python
# (c) 2020 slegroux@ccrma.stanford.edu

from data import Transcript, WavFile, get_basename
import pandas as pd
import glob, os
from torch.utils.data import Dataset
from pathlib import Path
from pandasql import sqldf
from IPython import embed
from torchaudio import transform


class DIMEX(Dataset):
    def __init__(self, root_path:str, transform=None):
        self._transform = transform
        audio_paths = Path(root_path).rglob('*/audio_editado/*/*.wav')
        self._audio_df = pd.DataFrame(list(audio_paths), columns=['path'])

        transcripts = Path(root_path).rglob('*/texto/*/*.utf8')
        self._transcript_df = pd.DataFrame(list(transcripts), columns=['path'])
        
        lambdas = {
           'shared': lambda x: x.parts[-2],
           'id': lambda x: x.stem,
           'sid': lambda x: x.parts[-4]
        }

        for i in ('shared', 'id', 'sid'):
            self._audio_df[i] = self._audio_df.path.apply(lambdas[i])
            self._transcript_df[i] = self._transcript_df.path.apply(lambdas[i])

        # Path type not recognized by pandasql => convert to string
        audio_df = self._audio_df
        audio_df.path = audio_df.path.astype(str)
        transcript_df = self._transcript_df
        transcript_df.path = transcript_df.path.astype(str)

        # join tables by id & shared 
        q = "select a.sid, a.id, a.shared, a.path as audio_path, t.path as transcript_path \
            from audio_df a join transcript_df t on a.id = t.id and a.shared = t.shared;"

        self._ds = sqldf(q, locals())


    def __getitem__(self, n):
        audio_path = str(self._ds.iloc[n].audio_path)
        trn_path = str(self._ds.iloc[n].transcript_path)
        sid = self._ds.iloc[n].sid
        shared = self._ds.iloc[n].shared
        id = self._ds.iloc[n].id
        uid = sid + '_' + id
        if self.transform:
            transformed = 
        w = WavFile(audio_path, language='es', dialect='MX')
        t = Transcript(trn_path, language='es', dialect='MX', encoding='utf-8')
        return(uid, w.waveform, w.sr, w.duration, t.transcript)

    def __len__(self):
        return(len(self._audios))
    
    @property
    def audio_df(self):
        return(self._audio_df)
    
    @property
    def transcript_df(self):
        return(self._transcript_df)

