#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob, os
import wave
from abc import ABC, abstractmethod
import uuid
from pandasql import sqldf
from IPython import embed
import csv
import string
import spacy


def get_basename(filename:str):
    bn = os.path.splitext(os.path.basename(filename))
    if bn[1] == '':
        return(bn[0])
    else:
        return(get_basename(bn[0]))

""" def get_transcript(filename:str):
    with open(filename, 'r', encoding='utf-8') as f:
        return(f.read().strip())
 """


class Transcript(object):
    def __init__(self, path:str, language='english', dialect='american'):
        self._path = path
        self._language = language
        self._dialect = dialect
    
    @staticmethod
    def get_transcript(filename:str):
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
        return(self.get_transcript(self.path))


class Transcripts(object):
    def __init__(self, regex:str, tr_cls=None):
        self._file_list = glob.glob(regex)
        self._tr_cls = tr_cls
    
    @property
    def df(self):
        pass


class WavFile(object):
    def __init__(self, path:str, language='en', dialect='US', gender=None, suffix=''):
        self._path = path
        self._format = 'wav'
        self._language = language
        self._dialect = dialect
        self._gender = None
        self._suffix = suffix

    @property
    def language(self):
        return(self._language)

    @language.setter
    def language(self, language):
        self._language = language

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
        return( self.get_wav_info(self._path)[0])
    
    @property
    def duration(self):
        return( self.get_wav_info(self._path)[1])
    
    @staticmethod
    def get_wav_info(filename:str):
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
    def __init__(self, audio_path, tr_path, audio_cls, tr_cls, name='dataset', lang='english'):
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
        if lang=='spanish':
            self._nlp = spacy.load("es_core_news_sm")
        elif lang=='english':
            self._nlp = spacy.load("en_core_web_sm")
        
        # check if variable is actually defined anywhere
        if not (hasattr(self, '_csv_path')):
            self._df = self._get_joined_df()

    @classmethod
    def init_with_csv(cls, csv_path, ids, name='dataset', lang='english'):
        cls._csv_path = csv_path
        cls._ids = ids
        cls._df =cls._get_df_from_csv(cls, cls._ids)
        return(cls(None, None, None, None, name, lang))

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

    def _get_joined_df(self):
        wav_df = self.wav
        tr_df = self.tr
        q_ans = self.query.format("wav_df", "tr_df")        
        joined = sqldf(q_ans, locals())
        self.add_uuid(joined)
        self.add_table_name(joined, self._name)
        return(joined)
    
    def _get_df_from_csv(self, ids, sep='\t', header=0, name='common_voice'):
        
        df = pd.read_csv(self._csv_path, sep=sep, header=header, names=ids)
        self.add_uuid(df)
        self.add_table_name(df, name)
        df['transcript_path'] = [os.path.abspath(self._csv_path)] * len(df)        
        df['duration'] = df['audio_path'].apply(lambda x: WavFile.get_wav_info(x)[1])
        self._df = df
        return(df)

    @property
    def df(self):
        return(self._df)
    
    @staticmethod
    def add_uuid(df):
        df['uuid'] = [ uuid.uuid4() for i in range(len(df))] 

    @staticmethod
    def add_table_name(df, name):
        df['dataset_id'] = [name] * len(df)
    
    def pickle(self, path):
        self.df.to_pickle(path)

    def remove_punc(self, sentence:str)-> str:
        # words = sentence.split(' ')
        # table = str.maketrans('', '', string.punctuation)
        # stripped = [w.translate(table) for w in words]
        # no_white = [s for s in stripped if s]

        doc = self._nlp(sentence)
        res = [(w.text, w.pos_) for w in doc]
        return(' '.join([w.lower() for w,att  in res if att!= 'PUNCT']))
   

    def export2kaldi(self, dir_path, lang='english'):
        try:
            os.mkdir(dir_path)
        except OSError as error:
            print(error)
        
        # kaldi needs uuid that starts by sid for sorting
        # http://kaldi-asr.org/doc/data_prep.html
        # convert uuid type to string to be able to add sid to it

        self._df['uuid'] = self._df['uuid'].apply(lambda x: x.urn[9:])
        self._df['uuid'] = self._df['sid'] + '_' + self._df['uuid']
        wav_scp = self.df[['uuid', 'audio_path']]
        wav_scp.to_csv(dir_path + '/wav.scp', sep=' ', index=False, header=None)
        utt2spk = self.df[['uuid','sid']]
        utt2spk.to_csv(dir_path + '/utt2spk', sep=' ', index=False, header=None)
        self._df['transcript'] = self.df['transcript'].apply(lambda x: self.remove_punc(x))
        text = self._df[['uuid','transcript']]


        try:
            text.to_csv(dir_path + '/text', sep=' ', index=False, header=None)
        except IOError:
            print("File already exists. Delete or change path")


if __name__ == "__main__":
    pass    
