#!/usr/bin/env python
# (c) 2020 slegroux@ccrma.stanford.edu

from data import Transcript, WavFile, SpeechDataset, TextNormalizer
import pandas as pd
from pathlib import Path
from pandasql import sqldf
from IPython import embed
import torchaudio


class DIMEX(SpeechDataset):
    def __init__(self, root_path:str, resample:int=None, normalize:bool=False):
        self._language = 'es'
        self._dialect = 'MX'
        self._resample = resample
        self._normalize = normalize
        self._normalizer = None
        self._ds = None
        if self._normalize:
            self._normalizer = TextNormalizer(language=self._language)
        
        if not self._ds:
            # generate ds dataframe from folders in case it isn't provided directly    
            audio_paths = Path(root_path).absolute().rglob('*/audio_editado/*/*.wav')
            self._audio_df = pd.DataFrame(list(audio_paths), columns=['path'])

            transcripts = Path(root_path).absolute().rglob('*/texto/*/*.utf8')
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
            ds = sqldf(q, locals())
            ds['uid'] = ds['sid'] + '_' + ds['id'] + '_' + ds['shared']
            self._ds = ds
        
        self._length = len(self._ds)
    
    @classmethod
    def init_from_df(cls, df):
        self._ds = df
        return()

    def __getitem__(self, n):
        audio_path = str(self._ds.iloc[n].audio_path)
        trn_path = str(self._ds.iloc[n].transcript_path)
        uid = str(self._ds.iloc[n].uid)
        w = WavFile(audio_path, language='es', dialect='MX')
        if self._resample:
            w.waveform = torchaudio.transforms.Resample(w.sr, self._resample)(w.waveform)
            w.sr = self._resample
        t = Transcript(trn_path, language='es', dialect='MX', encoding='utf-8', normalizer=self._normalizer)
        return(uid, w.waveform, w.sr, w.duration, t.transcript)

    def __len__(self):
        return(self._length)
    
    def export2kaldi(self, path):
        super().export2kaldi(path, language='es', dialect='MX', normalizer=self._normalizer, resample=self._resample)

    @property
    def ds(self):
        return(self._ds)

    

