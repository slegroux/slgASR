#!/usr/bin/env python

from data import WavFile, WavFiles, Transcript, get_basename
import pandas as pd

class HeroicoTranscripts(Transcript):
    def __init__(self, path):
        Transcript.__init__(self, path, lang='es', dialect='mexican', encoding='ISO-8859-1')
    
    @property
    def df(self):
        df = pd.read_csv(self._path, encoding='ISO-8859-1', sep='\t', header=None, names=['uid', 'transcript'])
        df['language'] = [self._lang]*len(df)
        df['dialect'] = [self._dialect]*len(df)
        df['path'] = [self._path]*len(df)
        if df['uid'].dtype == 'object':
            new = df['uid'].str.split('/',expand=True)
            df['sid'] = new[0]
            df['uid_temp'] = new[1]
            df.drop(columns=['uid'], inplace=True)
            df.rename(columns={'uid_temp':'uid'},inplace=True)
        return(df)


class HeroicoWavFile(WavFile):
    def __init__(self, path:str, suffix=''):
        WavFile.__init__(self, path, lang='spanish', dialect='mexican', suffix=suffix)
    
    def _get_sid(self):
        return(self._path.split('/')[-2])

    @property
    def uid(self):
        uid = get_basename(self._path)
        return(uid)
    
    @property
    def sid(self):
        return self._get_sid()



