#!/usr/bin/env python
from data import Transcript, WavFile, get_basename
import pandas as pd
import glob, os


class DimexTranscript(Transcript):
    def __init__(self, path:str):
        Transcript.__init__(self, path)
        self._language = 'spanish'
        self._dialect = 'mexican'
        self._basename = get_basename(path)

    @property
    def language(self):
        return(self._language)
    
    @property
    def dialect(self):
        return(self._dialect)

    @property
    def uid(self):
        return(self._basename)

    @property
    def sid(self):
        return(self._basename[:4])
    
    @property
    def info(self):
        return    


class DimexTranscripts(DimexTranscript):
    def __init__(self, regex:str):
        self._file_list = glob.glob(regex)

    def _process_transcripts(regex:str, language:str, suffix:str=''):
        paths = [ (get_basename(x) + suffix, get_basename(x)[:4], os.path.abspath(x), Transcript.get_transcript(os.path.abspath(x)), language) for x in glob.glob(regex)]
        df = pd.DataFrame(paths, columns=['uid', 'sid', 'path', 'transcript', 'language'])
        return(df)
    
    @property
    def df(self):
        d = lambda x: DimexTranscript(x)
        paths = [  (d(x).uid, d(x).sid, d(x).path, d(x).transcript, d(x).language) for x in self._file_list]    
        df = pd.DataFrame(paths, columns=['uid', 'sid', 'path', 'transcript', 'language'])
        return(df)


class DimexWavFile(WavFile):
    def __init__(self, path:str):
        WavFile.__init__(self, path)
        self._language = 'spanish'
        self._dialect = 'mexican'
        self._basename = get_basename(path)

    @property
    def language(self):
        return(self._language)
    
    @property
    def dialect(self):
        return(self._dialect)

    @property
    def uid(self):
        return(self._basename)

    @property
    def sid(self):
        return(self._basename[:4])
    
    @property
    def info(self):
        return


""" class DimexSpeechFiles(DimexSpeechFile):
    def __init__(self, regex:str):
        self._file_list = glob.glob(regex)
    
    @property
    def df(self):
        d = lambda x: DimexSpeechFile(x)
        paths = [  (d(x).uid, d(x).sid, d(x).path, d(x).sr, d(x).duration, d(x).format, d(x).language, d(x).dialect) for x in self._file_list]    
        df = pd.DataFrame(paths, columns=['uid', 'sid', 'path', 'sr', 'duration', 'format', 'language','dialect'])
        return(df) """
        