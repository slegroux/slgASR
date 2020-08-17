#!/usr/bin/env python
# to run tests:
# pytest -p no:warnings -svDATA_FOLDER +'.py

import pytest
# from data import get_basename, Transcript, WavFile, WavFiles, TextNormalizer, ASRDataset
# from data_dimex import DIMEX
# from data_heroico import HeroicoTranscripts, HeroicoWavFile
# from data_common_voice import CommonVoiceDF
from data import TextNormalizer, Audio, Transcript, Audios, Transcripts
from data import ASRDataset, ASRDatasetCSV
import numpy as np
from IPython import embed
from pathlib import Path

DATA_FOLDER='data/tests'

def test_text_normalizer():
    normalizer = TextNormalizer()
    assert normalizer.normalize('¿ HEY!') == '¿ hey'
    normalizer = TextNormalizer(lang='es')
    assert normalizer.normalize('¿Hola!') == 'hola'

# fixture to init global variables
@pytest.fixture(scope="module")
def data_():
    data = {
        'root': DATA_FOLDER +'/dimex100',
        'wavfile': DATA_FOLDER + '/dimex100/s058/audio_editado/comunes/s05810.wav',
        'mp3file': DATA_FOLDER + '/dimex100/s058/audio_editado/comunes/s05810.mp3',
        'transcript': DATA_FOLDER + '/dimex100/s058/texto/comunes/s05810.utf8',
        'transcripts': DATA_FOLDER + '/dimex100/s058/texto/comunes/*.utf8',
        'audios': DATA_FOLDER + '/dimex100/s058/audio_editado/comunes/*.wav'
        }
    return data

def test_wav(data_):    
    w = Audio(data_['wavfile'])
    w.lang = 'fr'
    w.country = 'CA'
    assert (w.path, w.sr, w.duration, w.lang, w.country) == \
        (str(Path(data_['wavfile']).absolute()), 16000, 3.5403125, 'fr', 'CA')

def test_mp3(data_):    
    w = Audio(data_['mp3file'])
    w.lang = 'fr'
    w.country = 'CA'
    assert (w.path, w.sr, w.duration, w.lang, w.country) == \
        (str(Path(data_['mp3file']).absolute()), 16000, 3.636, 'fr', 'CA')

def test_transcript(data_):
    t = Transcript(data_['transcript'], lang='es', normalize=True, encoding='utf-8')
    t.lang = 'es'
    t.country = 'MX'
    trans = t.text
    assert (trans == u"recopilación de firmas en contra de la extrema derecha de austria")
    assert Path(t.path) == Path(data_['transcript']).resolve()
    assert (t.encoding, t.lang, t.country) == ('utf-8', 'es', 'MX')

def test_transcripts(data_):
    # case where wav & trn ids are filename
    t = Transcripts(data_['transcripts'], normalize=True, lang='es', country='MX')
    trn = t.transcripts
    assert trn.text[0] == u"recopilación de firmas en contra de la extrema derecha de austria"
    assert trn.path[0] == '/Users/syl20/Projects/slgASR/data/tests/dimex100/s058/texto/comunes/s05810.utf8'
    assert (trn.lang[0], trn.country[0]) == ('es', 'MX')
    assert trn.id[0] == 's05810'
    assert trn.encoding[0] == 'utf-8'

def test_audios(data_):
    a = Audios(data_['audios'], lang='es', country='MX')
    aa = a.audios
    assert Path(aa.path[0]) == Path(data_['wavfile']).resolve()
    assert aa.sr[0] == 16000
    assert aa.duration[0] == 3.5403125
    assert aa.lang[0] == 'es'
    assert aa.country[0] == 'MX'

def test_asr_dataset(data_):
    t = Transcripts(data_['transcripts'], normalize=True, lang='es', country='MX')
    a = Audios(data_['audios'], lang='es', country='MX')
    ds = ASRDataset(a.audios, t.transcripts)
    ds.export2kaldi('/tmp/kaldi_dimex')

@pytest.fixture(scope="module")
def common_voice_data():
    data = {
        'path': 'data/tests/common_voice/test.tsv',
        'audio_path': 'data/tests/common_voice/clips_16k'
        }
    return(data)

def test_get_df_from_csv(common_voice_data):
    ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']
    ds = ASRDataset.init_with_csv(common_voice_data['path'], ids, name='common_voice', lang='es', prepend_audio_path='', normalize=True)
    assert ds.df.iloc[0].transcript == 'pero en un lugar para nosotros solos'

def test_common_voice_df(common_voice_data):
    cv = CommonVoiceDF(common_voice_data['path'])
    assert cv.df.duration[0] == np.float64(3.168)

def test_to_kaldi(common_voice_data):
    ids = ['sid', 'audio_path', 'transcript', 'up_votes', 'down_votes', 'age', 'gender', 'dialect']
    ds = ASRDataset.init_with_csv(common_voice_data['path'], ids, name='common_voice', prepend_audio_path='/Users/syl20/Projects/slgASR')
    ds.export2kaldi('/tmp/kaldi_dir')
    
