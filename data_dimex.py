#!/usr/bin/env python
# (c) 2020 slegroux@ccrma.stanford.edu

from data import Transcript, WavFile, SpeechDataset, TextNormalizer, ASRDataset
import pandas as pd
from pathlib import Path
from pandasql import sqldf
import torchaudio
import uuid
import spacy


class DIMEX(ASRDataset):
    def __init__(self, root_path:str, resample:int=None, normalize:bool=False):
        self._lang = 'es'
        self._nlp = spacy.load("es_core_news_sm")
        self._dialect = 'MX'
        self._resample = resample
        self._normalize = normalize

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
        df = sqldf(q, locals())
        # ds['uuid'] = ds['sid'] + '_' + ds['id'] + '_' + ds['shared']
        df['uuid'] = [ uuid.uuid4() for i in range(len(df))]
        df['transcript'] = df['transcript_path'].apply(lambda x: open(x).read().strip())
        self._df = df
        self._length = len(self._df)
    
    @classmethod
    def init_from_df(cls, df):
        cls._df = df
        return()

    def __getitem__(self, n):
        audio_path = str(self._df.iloc[n].audio_path)
        trn_path = str(self._df.iloc[n].transcript_path)
        uuid = str(self._df.iloc[n].uuid)
        w = WavFile(audio_path, lang='es', dialect='MX')
        if self._resample:
            w.waveform = torchaudio.transforms.Resample(w.sr, self._resample)(w.waveform)
            w.sr = self._resample
        t = Transcript(trn_path, lang='es', dialect='MX', encoding='utf-8', normalize=self._normalize)
        return(uuid, w.waveform, w.sr, w.duration, t.transcript)

    def __len__(self):
        return(self._length)
    
    def export2kaldi(self, path):
        super().export2kaldi(path, lang='es',sr=self._resample)

    @property
    def df(self):
        return(self._df)

    

