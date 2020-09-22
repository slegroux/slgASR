#!/usr/bin/env python
# (c) 2020 slegroux@ccrma.stanford.edu

from data import DatasetSplit
from data_dimex import DIMEX
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DIMEX data prep for Kaldi")
    parser.add_argument('input_dir', help="data root dir")
    parser.add_argument('output_dir', help="data output dir")
    parser.add_argument('-r', '--resample', type=int, default=None)
    parser.add_argument('-n', '--normalize', type=bool, default=False)
    args = parser.parse_args()

    dimex = DIMEX(args.input_dir, resample=args.resample, normalize=args.normalize)
    #dimex.export2kaldi(args.output_dir)

    splitter = DatasetSplit(dimex)
    tr_df, tst_df = splitter.split()
    tr = DIMEX.init_from_df(tr_df)
    tst = DIMEX.init_from_df(tst_df)
    embed()
    train.export2kaldi('/tmp/train')
