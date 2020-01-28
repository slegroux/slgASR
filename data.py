#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob, os
import wave
from abc import ABC, abstractmethod
import uuid
from pandasql import sqldf
from IPython import embed

def get_basename(filename:str):
    bn = os.path.splitext(os.path.basename(filename))
    if bn[1] == '':
        return(bn[0])
    else:
        return(get_basename(bn[0]))

def get_transcript(filename:str):
    with open(filename, 'r', encoding='utf-8') as f:
        return(f.read().strip())


class Transcript(object):
    def __init__(self, path:str, language='english', dialect='american'):
        self._path = path
        self._language = language
        self._dialect = dialect

    def _get_transcript(self, filename:str):
        with open(filename, 'r', encoding='utf-8') as f:
            return(f.read().strip())

    @property
    def path(self):
        return(self._path)
    
    @property
    def language(self):
        return(self._language)
    
    @property
    def dialect(self):
        return(self._dialect)

    @property
    def transcript(self):
        return(self._get_transcript(self.path))


class Transcripts(object):
    def __init__(self, regex:str, tr_cls=None):
        self._file_list = glob.glob(regex)
        self._tr_cls = tr_cls
    
    @property
    def df(self):
        pass


class WavFile(object):
    def __init__(self, path:str, language='english', dialect='american', gender=None, suffix=''):
        self._path = path
        self._format = 'wav'
        self._language = language
        self._dialect = dialect
        self._gender = None
        self._suffix = suffix

    @property
    def language(self):
        return(self._language)
    
    @property
    def dialect(self):
        return(self._dialect)
    
    @property
    def gender(self):
        return(self._gender)
    
    @property
    def uid(self):
        pass
    
    @property
    def path(self):
        return(self._path)
    
    @property
    def format(self):
        return(self._format)

    @property
    def sr(self):
        return( self._get_wav_info(self._path)[0])
    
    @property
    def duration(self):
        return( self._get_wav_info(self._path)[1])
    
    def _get_wav_info(self, filename:str):
        with wave.open(filename, 'r') as f:
            n_frames = f.getnframes()
            frame_rate = f.getframerate()
            duration  = n_frames / float(frame_rate)
        return(frame_rate, duration)


class WavFiles(object):
    def __init__(self, regex, wf_cls=None, suffix=''):
        self._file_list = glob.glob(regex)
        self._wf_cls = wf_cls
        self._suffix = suffix
    
    @property
    def df(self):
        d = lambda x: self._wf_cls(x)
        paths = [(d(x).uid, d(x).sid, d(x).path, d(x).sr, \
            d(x).duration, d(x).format, d(x).language, d(x).dialect) for x in self._file_list]    
        df = pd.DataFrame(paths, columns=['uid', 'sid', 'path', \
            'sr', 'duration', 'format', 'language','dialect'])
        return(df)


class ASRDataset(object):
    def __init__(self, audio_path, tr_path, audio_cls, tr_cls, name='dataset'):
        DEFAULT_QUERY = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid and {0}.sid={1}.sid"
        self._audio_cls = audio_cls
        self._tr_cls = tr_cls
        self._audio_path = audio_path
        self._tr_path = tr_path
        self._wav = None
        self._tr = None
        self._name = name
        self._query = DEFAULT_QUERY
    
    @property
    def wav(self):
        return(WavFiles(self._audio_path, self._audio_cls).df)

    @property
    def tr(self):
        return(self._tr_cls(self._tr_path).df)

    @property
    def query(self):
        return(self._query)
    
    @query.setter
    def query(self, query):
        self._query = query

    @property
    def df(self):
        wav_df = self.wav
        tr_df = self.tr
        q_ans = self.query.format("wav_df", "tr_df")        
        joined = sqldf(q_ans, locals())
        self.add_uuid(joined)
        self.add_table_name(joined, self._name)
        return(joined)

    def add_uuid(self, df):
        df['uuid'] = [ uuid.uuid4() for i in range(len(df))] 

    def add_table_name(self, df, name):
        df['dataset_id'] = [name] * len(df)
    
    def pickle(self, path):
        self.df.to_pickle(path)


class DataFactory(ABC):
    @abstractmethod
    def create_wavfile(self):
        pass
    
    @abstractmethod
    def create_transcripts(self):
        pass




if __name__ == "__main__":
    pass    
