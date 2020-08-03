#!/usr/bin/env python
# (c) 2020 slegroux@ccrma.stanford.edu

import pandas as pd
import numpy as np
import glob, os
import wave
from abc import ABC, abstractmethod
import uuid
from pandasql import sqldf
import csv
import string
import spacy
import sys
import torchaudio
from torch.utils.data import Dataset, random_split, DataLoader
from IPython import embed


def get_basename(filename:str):
    bn = os.path.splitext(os.path.basename(filename))
    if bn[1] == '':
        return(bn[0])
    else:
        return(get_basename(bn[0]))


class TextNormalizer(object):
    def __init__(self, language:str='en'):
        #TODO check for other languages & add other normalization rules
        if language=='es':
            self._nlp = spacy.load("es_core_news_sm")
        elif language=='en':
            self._nlp = spacy.load("en_core_web_sm")
        self._text = None
        
    def normalize(self, text:str)->str:
        self._text = self.remove_punc(text)
        return(self._text)

    def remove_punc(self, sentence:str)-> str:
        doc = self._nlp(sentence)
        res = [(w.text, w.pos_) for w in doc]
        return(' '.join([w.lower() for w,att  in res if att!= 'PUNCT']))
    
    @property
    def text(self):
        return(self._text)


class Transcript(object):
    def __init__(self, path:str, language='en', dialect='US', encoding='utf-8', normalizer=None):
        self._path = path
        self._language = language
        self._dialect = dialect
        self._encoding = encoding
        if normalizer:
            text = self.get_transcript(self.path)
            self._transcript = normalizer.normalize(text)
        else:
            self._transcript = self.get_transcript(self.path)
        
    def get_transcript(self, filename:str)->str:
        with open(filename, 'r', encoding=self._encoding) as f:
            return(f.read().strip())

    @property
    def path(self):
        return(self._path)
    
    @property
    def encoding(self):
        return(self._encoding)
    
    @encoding.setter
    def encoding(self, enc):
        self._encoding = enc
    
    @property
    def language(self):
        return(self._language)
    
    @language.setter
    def language(self, lang):
        self._language = lang
    
    @property
    def dialect(self):
        return(self._dialect)
    
    @dialect.setter
    def dialect(self, dialect):
        self._dialect = dialect

    @property
    def transcript(self):
        return(self._transcript)
    
    # for e.g. normalization of transcript
    @transcript.setter
    def transcript(self, trn):
        sef._transcript = trn


class WavFile(object):
    def __init__(self, path:str, language='en', dialect='US', gender=None, suffix=''):
        self._path = path
        self._format = 'wav'
        self._language = language
        self._dialect = dialect
        self._gender = None
        self._suffix = suffix
        self._uuid = str(uuid.uuid4())
        self._waveform, self._sample_rate = torchaudio.load(self._path)
    
    @property
    def language(self):
        return(self._language)

    @language.setter
    def language(self, language):
        self._language = language

    @property
    def dialect(self):
        return(self._dialect)

    @dialect.setter
    def dialect(self, dialect):
        self._dialect = dialect
    
    @property
    def gender(self):
        return(self._gender)
    
    @gender.setter
    def gender(self, gender):
        self._gender = gender
    
    @property
    def uuid(self):
        return(self._uuid)
    
    @property
    def path(self):
        return(self._path)
    
    @property
    def format(self):
        return(self._format)

    @property
    def sr(self):
        return( self._sample_rate)
    
    # in case changed by a transform such as Resample
    @sr.setter
    def sr(self, sr):
        self._sample_rate = sr
    
    @property
    def duration(self):
        return( self.get_wav_info(self._path)[1])

    @property
    def waveform(self):
        return(self._waveform)
    
    # for transforms (like resample and such)
    @waveform.setter
    def waveform(self, waveform):
        self._waveform = waveform
    
    @staticmethod
    def get_wav_info(filename:str):
        try:
            info = torchaudio.info(filename)
            duration = info[0].length / info[0].rate
            return(info[0].rate, duration)
        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            return(None,None)


class SpeechDataset(Dataset):
    def __init__(self):
        self._ds = None
        self._train = None
        self._test = None
    
    @classmethod
    def init_from_df(cls, df):
        pass

    @classmethod
    def init_from_csv(self, path:str):
        pass
    
    def __getitem__(self):
        pass

    def __len__(self):
        pass

    def export2kaldi(self, dir_path:str, language:str='en', \
        dialect:str='US', encoding:str='utf-8', normalizer=None, resample:int=None):
        try:
            os.mkdir(dir_path)
        except OSError as error:
            print(error)

        # kaldi needs uuid that starts by sid for sorting
        # http://kaldi-asr.org/doc/data_prep.html
        # convert uuid type to string to be able to add sid to it
        # TODO online resampling + take care of text normalization

        if resample:
            sox_resample = lambda x: "sox " + x + " -t wav -r " + str(resample) + " -c 1 - |"
            wav_scp = self._ds[['uid', 'audio_path']]
            soxed = wav_scp['audio_path'].apply(sox_resample).copy()
            # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
            wav_scp = wav_scp.copy()
            wav_scp['audio_path'] = soxed

        else:
            wav_scp = self._ds[['uid', 'audio_path']] 
        wav_scp.to_csv(dir_path + '/wav.scp', sep=' ', index=False, header=None)
        
        utt2spk = self._ds[['uid','sid']]
        utt2spk.to_csv(dir_path + '/utt2spk', sep=' ', index=False, header=None)

        self._ds['transcript'] = self._ds['transcript_path'].apply(lambda x: Transcript(x, language=language, dialect=dialect, encoding=encoding, normalizer=normalizer).transcript)
        text = self._ds[['uid','transcript']]

        try:
            text.to_csv(dir_path + '/text', sep=' ', index=False, header=None)
        except IOError:
            print("File already exists. Delete or change path")


class DatasetSplit(object):
    def __init__(self, dataset, split:float=0.8, shuffle=False):
        self._split = split
        self._shuffle = shuffle
        self._ds = dataset._ds
        self._train = None
        self._test = None

    def split(self):
        data_len = len(self._ds)
        train_len = int(self._split * data_len)
        test_len = data_len - train_len

        if self._shuffle:
            self._ds = self._ds.sample(len(self._ds))
        self._train = self._ds[:train_len]
        self._test = self._ds[train_len:]
        return(self._train, self._test)


class Transcripts(object):
    def __init__(self, regex:str, tr_cls=None):
        self._file_list = glob.glob(regex)
        self._tr_cls = tr_cls
    
    @property
    def df(self):
        pass


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

DEFAULT_QUERY = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid and {0}.sid={1}.sid"

class ASRDataset(object):
    def __init__(self, audio_path, tr_path, audio_cls, tr_cls, name='dataset', lang='en', query=DEFAULT_QUERY):

        self._audio_cls = audio_cls
        self._tr_cls = tr_cls
        self._audio_path = audio_path
        self._tr_path = tr_path
        self._wav = None
        self._tr = None 
        self._name = name
        self._query = query
        if lang=='es':
            self._nlp = spacy.load("es_core_news_sm")
        elif lang=='en':
            self._nlp = spacy.load("en_core_web_sm")
        
        # check if variable is actually defined anywhere
        if not (hasattr(self, '_csv_path')):
            self._df = self._get_joined_df()

    @classmethod
    def init_with_csv(cls, csv_path, ids, name='dataset', lang='en', prepend_audio_path=''):
        cls._csv_path = csv_path
        cls._ids = ids
        cls._df =cls._get_df_from_csv(cls, cls._ids, prepend_audio_path=prepend_audio_path)
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

    def _get_df_from_csv(self, ids, sep='\t', header=0, name='common_voice', prepend_audio_path=''):
        df = pd.read_csv(self._csv_path, sep=sep, header=header, names=ids)
        self.add_uuid(df)
        self.add_table_name(df, name)
        if prepend_audio_path:
            df['audio_path'] = prepend_audio_path + '/' + df['audio_path']
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
   

    def export2kaldi(self, dir_path, lang='english', sr=16000):
        try:
            os.mkdir(dir_path)
        except OSError as error:
            print(error)
        
        # kaldi needs uuid that starts by sid for sorting
        # http://kaldi-asr.org/doc/data_prep.html
        # convert uuid type to string to be able to add sid to it

        self._df['uuid'] = self._df['uuid'].apply(lambda x: x.urn[9:])
        self._df['uuid'] = self._df['sid'] + '_' + self._df['uuid']
        # hard copy otherwise it's just a view and then cannot reassign col values
        wav_scp = self.df[['uuid', 'audio_path']].copy()
        wav_scp.audio_path = 'sox ' + wav_scp.audio_path + ' -t wav -r ' + str(sr) + ' -c 1 -b 16 - |'
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
