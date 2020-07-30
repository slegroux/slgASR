#!/usr/bin/env python

from data import ASRDataset
from data_heroico import HeroicoWavFile, HeroicoTranscripts
from data_dimex import DimexWavFile, DimexTranscripts


## HEROICO
## answers
tr_answers = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-answers.txt"
wav_answers = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/heroico/Answers_Spanish/*/*.wav"
q = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid and {0}.sid={1}.sid"

answers = ASRDataset(wav_answers, tr_answers, HeroicoWavFile, HeroicoTranscripts, 'answers')
answers.query = q
answers.pickle('heroico_answers.pkl')



## Recordings
tr_recordings = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-recordings.txt"
wav_recordings = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/heroico/Recordings_Spanish/*/*.wav"
q_rec = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
        from {0} join {1} on {0}.uid={1}.uid"

recordings = ASRDataset(wav_recordings, tr_recordings, HeroicoWavFile, HeroicoTranscripts, 'recordings')
recordings.query = q_rec
recordings.pickle('heroico_recordings.pkl')


# USMA
usma = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/usma-prompts.txt'
native_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/native*/*.wav'
nonnative_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/nonnative*/*.wav'


# DIMEX
tr_common = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/comunes/*.txt.utf8'
wav_common = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/comunes/*.wav'
comunes = ASRDataset(wav_common, tr_common, DimexWavFile, DimexTranscripts, 'comunes')


tr_individuales = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/individuales/*.txt.utf8'
wav_individuales = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/individuales/*.wav'
individuales = ASRDataset(wav_individuales, tr_individuales, DimexWavFile, DimexTranscripts, 'individuales')

comunes.pickle('dimex_common.pkl')
individuales.pickle('dimex_individual.pkl')

embed()




