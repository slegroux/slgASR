#!/usr/bin/env python

import pytest
from data import get_basename
from data import WavFile, DimexSpeechFile, DimexSpeechFiles
from data import Transcript, Transcripts
from IPython import embed

@pytest.fixture(scope="module")
def data_():
    data = {
        'wavfile': '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/s058/audio_16k/comunes/s05810.wav',
        'regex_all_wavs': '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/*/*.wav',
        'transcript': '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/s100/texto/individuales/s10001.txt.utf8',
        'regex_all_transcripts': None
        }
    return data

def test_get_basename():
    bn = get_basename('/toto/test.txt.utf.totot')
    assert bn == 'test'

def test_wav_file(data_):    
    w = WavFile(data_['wavfile'])
    assert (w.path, w.sr, w.duration) == (data_['wavfile'], 16000, 3.5403125)

def test_dimex_file(data_):
    d = DimexSpeechFile(data_['wavfile'])
    assert (d.uid, d.sid, d.path, d.sr, d.duration, d.format, d.language, d.dialect) == \
        ('s05810_c', 's058', '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/s058/audio_16k/comunes/s05810.wav',
         16000, 3.5403125, 'wav', 'spanish', 'mexican')

def test_dimex_files(data_):
    dimex_speech_df = DimexSpeechFiles(data_['regex_all_wavs']).df


def test_transcript(data_):
    t = Transcript(data_['transcript']).transcript
    assert (t == u"Todos los productos y publicaciones de \"Adobe\" son de naturaleza comercial .")