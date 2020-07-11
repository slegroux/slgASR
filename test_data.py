#!/usr/bin/env python
# to run tests:
# pytest -p no:warnings -svDATA_FOLDER +'.py

import pytest
from data import get_basename, Transcript, WavFile, TextNormalizer
from data_dimex import DIMEX
from data_heroico import HeroicoTranscripts, HeroicoWavFile
from data_common_voice import CommonVoiceDF
import numpy as np
from IPython import embed

DATA_FOLDER='data/tests'

# fixture to init global variables
@pytest.fixture(scope="module")
def data_():
    data = {
        'root': DATA_FOLDER +'/dimex100',
        'wavfile': DATA_FOLDER + '/dimex100/s058/audio_editado/comunes/s05810.wav',
        'transcript': DATA_FOLDER + '/dimex100/s058/texto/comunes/s05810.utf8'
        }
    return data

# test WavFile class
def test_get_basename():
    bn = get_basename('/toto/test.txt.utf.totot')
    assert bn == 'test'

def test_wav_file(data_):    
    w = WavFile(data_['wavfile'])
    w.language = 'fr'
    w.dialect = 'CA'
    w.gender = 'M'
    assert (w.path, w.sr, w.duration, w.language, w.dialect, w.gender) == \
        (data_['wavfile'], 16000, 3.5403125, 'fr', 'CA', 'M')
    #TODO add test for waveform 

def test_text_normalizer(data_):
    normalizer = TextNormalizer()
    assert normalizer.normalize('Hey!') == 'hey'
    normalizer = TextNormalizer(language='es')
    assert normalizer.normalize('Hola!') == 'hola'

def test_transcript(data_):
    normalizer = TextNormalizer('es')
    t = Transcript(data_['transcript'], normalizer=normalizer)
    t.encoding = 'utf-8'
    t.language = 'es'
    t.dialect = 'MX'
    trans = t.transcript
    assert (trans == u"recopilación de firmas en contra de la extrema derecha de austria")
    assert t.path == data_['transcript']
    assert (t.encoding, t.language, t.dialect) == ('utf-8', 'es', 'MX')

def test_dimex(data_):
    dimex = DIMEX(data_['root'], resample=8000, normalize=True)
    assert (dimex[0][0], dimex[0][2], dimex[0][3], dimex[0][4]) == \
        ('s058_s05810_comunes',8000, 3.5403125,'recopilación de firmas en contra de la extrema derecha de austria')
    dimex = DIMEX(data_['root'], resample=8000, normalize=False)
    assert (dimex[0][0], dimex[0][2], dimex[0][3], dimex[0][4]) == \
        ('s058_s05810_comunes',8000, 3.5403125,'Recopilación de firmas en contra de la extrema derecha de Austria.')
    dimex.export2kaldi('/tmp/test2')


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
    
