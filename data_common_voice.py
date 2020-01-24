#!/usr/bin/env python
from data import Transcript, WavFile, get_basename, get_transcript
import pandas as pd
import glob, os


class CommonVoiceDF(object):
    def __init__(self, path):
        self._datapath = path
        self._df = pd.read_csv(self._datapath, del='\t')
    
    @property
    def df(self):
        return(self._df)
