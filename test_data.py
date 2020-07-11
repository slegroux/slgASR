#!/usr/bin/env python

# to run tests:
# pytest -p no:warnings -svDATA_FOLDER +'.py

import pytest
from data import get_basename, Transcript, WavFile, WavFiles, ASRDataset
from data_dimex import DimexWavFile, DimexTranscript, DimexTranscripts
from data_heroico import HeroicoTranscripts, HeroicoWavFile
from data_common_voice import CommonVoiceDF
import numpy as np
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

def test_heroico_join(heroico_data):
    
    q = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid"

    recordings = ASRDataset(heroico_data['regex_wavs_recordings'], heroico_data['transcript'], HeroicoWavFile, HeroicoTranscripts, query=q)
    assert recordings.df.transcript[0]  == 'iturbide se auto nombró generalísimo de mar y tierra'


@pytest.fixture(scope="module")
def common_voice_data():
    data = {
        'path': 'data/tests/common_voice/test.tsv'
        }
    return(data)

def test_get_df_from_csv(common_voice_data):
    ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']
    ds = ASRDataset.init_with_csv(common_voice_data['path'], ids, name='common_voice')
    assert ds.df.iloc[0].transcript == 'pero en un lugar para nosotros solos,'

def test_common_voice_df(common_voice_data):
    cv = CommonVoiceDF(common_voice_data['path'])
    assert cv.df.duration[0] == np.float64(3.168)

def test_to_kaldi(common_voice_data):
    ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']
    ds = ASRDataset.init_with_csv(common_voice_data['path'], ids, name='common_voice')

    ds.export2kaldi('/tmp/kaldi_dir')
    
