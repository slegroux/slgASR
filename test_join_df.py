#!/usr/bin/env python

from IPython import embed
from data import WavFiles
from data_heroico import HeroicoTranscripts, HeroicoWavFile
from data_dimex import DimexTranscripts, DimexWavFile

from pandasql import sqldf
import uuid

def add_uuid(df):
    df['uuid'] = [ uuid.uuid4() for i in range(len(df))] 

def get_wav_tr_df(transcripts, recordings, wav_class, trs_class):
    tr_df = trs_class(transcripts).df
    wav_df = WavFiles(recordings, wav_class).df
    return(wav_df, tr_df)

def build_join_query(wav_name_df:str, tr_name_df:str)->str:
    query = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid".format(wav_name_df, tr_name_df)
    return(query)

def join_df(wav_df_name, tr_df_name):
    query = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
            from {0} join {1} on {0}.uid={1}.uid".format(wav_df_name, tr_df_name)
    res = sqldf(query, globals())
    return(add_uuid(res))

## HEROICO
# transcripts
tr_answers = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-answers.txt"
tr_recordings = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/heroico-recordings.txt"
tr_answers_df = HeroicoTranscripts(tr_answers).df
tr_recordings_df = HeroicoTranscripts(tr_recordings).df

# audio
wav_recordings = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/heroico/Recordings_Spanish/*/*.wav"
wav_answers = "/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/heroico/Answers_Spanish/*/*.wav"
wav_recordings_df = WavFiles(wav_recordings, HeroicoWavFile).df
wav_answers_df = WavFiles(wav_answers, HeroicoWavFile).df

# tables
# for each uid (waveform) there is a different transcript
q_rec = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
        from {0} join {1} on {0}.uid={1}.uid".format("wav_recordings_df", "tr_recordings_df")

rec = sqldf(q_rec, locals())
add_uuid(rec)
rec.to_pickle('heroico_recordings.pkl')

# for each question (uid) there are multiple answers by different speakers
q_ans = "select {0}.uid, {0}.path as audio_path, {0}.sid, {0}.sr, {0}.duration, {0}.format, {0}.language, \
            {0}.dialect, {1}.path as transcript_path, {1}.transcript \
        from {0} join {1} on {0}.uid={1}.uid and {0}.sid={1}.sid".format("wav_answers_df", "tr_answers_df")

ans = sqldf(q_ans, locals())
add_uuid(ans)
ans.to_pickle('heroico_answers.pkl')

# USMA
usma = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/transcripts/usma-prompts.txt'
native_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/native*/*.wav'
nonnative_regex = '/home/workfit/Sylvain/Data/LDC/Heroico/LDC2006S37/data/speech/usma/nonnative*/*.wav'


# DIMEX
tr_reg_c = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/comunes/*.txt.utf8'
wav_reg_c = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/comunes/*.wav'
w_c, t_c  = get_wav_tr_df(tr_reg_c, wav_reg_c, DimexWavFile, DimexTranscripts)

tr_reg_i = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/*/individuales/*.txt.utf8'
wav_reg_i = '/home/workfit/Sylvain/Data/Spanish/CorpusDimex100/*/audio_16k/individuales/*.wav'
w_i, t_i  = get_wav_tr_df(tr_reg_i, wav_reg_i, DimexWavFile, DimexTranscripts)

q = build_join_query("w_c", "t_c")
dimex_common = sqldf(q, locals())
add_uuid(dimex_common)
dimex_common.to_pickle('dimex_common.pkl')

q = build_join_query("w_i", "t_i")
dimex_individual = sqldf(q, locals())
add_uuid(dimex_individual)
dimex_individual.to_pickle('dimex_individual.pkl')




