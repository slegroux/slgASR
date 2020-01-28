#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob, os
import wave
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


if __name__ == "__main__":
    pass    
