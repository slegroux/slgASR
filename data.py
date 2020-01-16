#!/usr/bin/env python

import pandas as pd
import numpy as np
import glob, os
import wave
from pandasql import sqldf
from IPython import embed

utterances = {'filename': None,
    'audio_path': None,
    'transcript_path': None,
    'transcript': None,
    'transcript_normalized': None,
    'sr': None,
    'audio_format': None,
    'sid': None,
    'gender': None,
    'duration': None,
    'language': None,
    'dialect': None
    }

lex = {
    'word': None,
    'phoneme': None,
    'grapheme': None
}

def get_basename(filename):
    bn = os.path.splitext(os.path.basename(filename))
    if bn[1] == '':
        return(bn[0])
    else:
        return(get_basename(bn[0]))

def get_transcript(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return(f.read().strip())

def get_wav_info(filename):
    with wave.open(filename, 'r') as f:
        n_frames = f.getnframes()
        frame_rate = f.getframerate()
        duration  = n_frames / float(frame_rate)
    return(frame_rate, duration)


def process_transcripts(regex, language, suffix=''):

    paths = [ (get_basename(x) + suffix, get_basename(x)[:4], os.path.abspath(x), get_transcript(os.path.abspath(x)), language) for x in glob.glob(regex)]
    df = pd.DataFrame(paths, columns=['uid', 'sid', 'path', 'transcript', 'language'])
    return(df)


def process_audio(regex, language, dialect=None, format='wav', suffix=''):
    uid = lambda x: get_basename(x) + suffix
    sid = lambda x: get_basename(x)[:4]
    path = lambda x: os.path.abspath(x)
    sr = lambda x: get_wav_info(os.path.abspath(x))[0]
    duration = lambda x: get_wav_info(os.path.abspath(x))[1]

    paths = [ (uid(x), sid(x), path(x), sr(x), duration(x), format, language, dialect) for x in glob.glob(regex)]
    df = pd.DataFrame(paths, columns=['uid', 'sid', 'path', 'sr', 'duration', 'format', 'language','dialect'])
    return(df)


if __name__ == "__main__":
    # basic tests
    print(get_basename('/toto/test.txt.utf.totot'))
    print(get_wav_info('/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/s058/audio_16k/comunes/s05810.wav')[0])

    # read heroico
    
    answers = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-answers.txt'
    recordings = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-recordings.txt'
    usma = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/usma-prompts.txt'

    native_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/native*/*.wav'
    nonnative_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/nonnative*/*.wav'
    
    # read dimex
    regex = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/comunes/*.txt.utf8'
    transcript_c = process_transcripts(regex, 'es', suffix='_c')
    regex = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/individuales/*.txt.utf8'
    transcript_i = process_transcripts(regex, 'es', suffix='_i')

    regex = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/comunes/*.wav'
    audio_c = process_audio(regex, 'es', 'mx', suffix='_c')
    regex = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/individuales/*.wav'
    audio_i = process_audio(regex, 'es', 'mx', suffix='_i')

    q_i = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
        from {0} join {1} on {0}.uid={1}.uid".format("audio_i", "transcript_i")
     
    q_c = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
        from {0} join {1} on {0}.uid={1}.uid".format("audio_c", "transcript_c")
    
    q_union = q_i + " union " + q_c
    
    individuales = sqldf(q_i, locals())
    comunes = sqldf(q_c, locals())
    union = sqldf(q_union, locals())
    print(union.head())
