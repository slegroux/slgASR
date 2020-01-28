#!/usr/bin/env python

# to run tests:
# pytest -p no:warnings -svDATA_FOLDER +'.py

import pytest
from data import get_basename, get_transcript, Transcript, WavFile, WavFiles
from data_dimex import DimexWavFile, DimexTranscript, DimexTranscripts
from data_heroico import HeroicoTranscripts, HeroicoWavFile
from IPython import embed

DATA_FOLDER='data/tests'

@pytest.fixture(scope="module")
def data_():
    data = {
        'wavfile': DATA_FOLDER +'/dimex100/audio_16k/comunes/s05810.wav',
        'regex_all_wavs': DATA_FOLDER +'/dimex100/audio_16k/*/*.wav',
        'transcript': DATA_FOLDER +'/dimex100/texto/comunes/s10001.txt.utf8',
        'regex_all_transcripts': DATA_FOLDER +'/dimex100/texto/*/*.txt.utf8'
        }
    return data

def test_get_basename():
    bn = get_basename('/toto/test.txt.utf.totot')
    assert bn == 'test'

def test_wav_file(data_):    
    w = WavFile(data_['wavfile'])
    assert (w.path, w.sr, w.duration) == (data_['wavfile'], 16000, 3.5403125)

def test_dimex_file(data_):
    d = DimexWavFile(data_['wavfile'])
    assert (d.uid, d.sid, d.path, d.sr, d.duration, d.format, d.language, d.dialect) == \
        ('s05810', 's058', data_['wavfile'],
         16000, 3.5403125, 'wav', 'spanish', 'mexican')

def test_dimex_files(data_):
    #dimex_speech_df = DimexSpeechFiles(data_['regex_all_wavs']).df
    dimex_speech_df = WavFiles(data_['regex_all_wavs'], DimexWavFile).df
    assert dimex_speech_df.iloc[0].values.tolist() == ['s05810', 's058', DATA_FOLDER +'/dimex100/audio_16k/comunes/s05810.wav', 16000, 3.5403125, 'wav', 'spanish', 'mexican']
    
def test_transcript(data_):
    t = Transcript(data_['transcript']).transcript
    assert (t == u"Todos los productos y publicaciones de \"Adobe\" son de naturaleza comercial .")

def test_dimex_transcript(data_):
    t = DimexTranscript(data_['transcript']).transcript
    assert (t == u"Todos los productos y publicaciones de \"Adobe\" son de naturaleza comercial .")

def test_dimex_transcripts(data_):
    dimex_transcripts_df = DimexTranscripts(data_['regex_all_transcripts']).df
    assert dimex_transcripts_df.iloc[0].values.tolist()  == ['s10001', 's100', DATA_FOLDER +'/dimex100/texto/comunes/s10001.txt.utf8', \
         'Todos los productos y publicaciones de "Adobe" son de naturaleza comercial .', 'spanish']


@pytest.fixture(scope="module")
def heroico_data():
    data = {
        'wavfile': DATA_FOLDER +'/heroico/speech/Recordings_Spanish/1/1.wav',
        'regex_wavs_recordings': DATA_FOLDER +'/heroico/speech/Recordings_Spanish/*/*.wav',
        'transcript': DATA_FOLDER +'/heroico/transcripts/heroico-recordings.txt'
        }
    return data

def test_heroico_transcripts(heroico_data):
    ht = HeroicoTranscripts(heroico_data['transcript']).df
    assert ht.iloc[0].transcript == 'iturbide se auto nombró generalísimo de mar y tierra'

def test_heroico_wav_file(heroico_data):
    hfw = HeroicoWavFile(heroico_data['wavfile'])
    assert (hfw.uid, hfw.sid, hfw.path) == ('1', '1', heroico_data['wavfile'])

def test_heroico_wav_files(heroico_data):
    heroico_df= WavFiles(heroico_data['regex_wavs_recordings'], HeroicoWavFile).df
    assert heroico_df.iloc[0].values.tolist() == ['1', '1', \
        DATA_FOLDER +'/heroico/speech/Recordings_Spanish/1/1.wav', 22050, \
        1.9127437641723355, 'wav', 'spanish','mexican']


@pytest.fixture(scope="module")

def common_voice_data():
    data = {
        'df': 'toto'
        }