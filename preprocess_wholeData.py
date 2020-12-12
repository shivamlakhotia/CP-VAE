import json
import numpy as np
import argparse
import os
import pandas as pd
import config

def concat_files(pth0, pth1, outpth, with_label=True):
    with open(outpth, "w") as f_out:
        with open(pth0, errors='ignore') as f0:
            for line in f0.readlines():
                if with_label:
                    f_out.write("0\t")
                f_out.write(line.strip() + "\n")
        with open(pth1, errors='ignore') as f1:
            for line in f1.readlines():
                if with_label:
                    f_out.write("1\t")
                f_out.write(line.strip() + "\n")

def main(args):
    data_pth = "data/%s" % args.data_name
    res_pth = "results/%s" % args.data_name
    for split in ["train","dev","test"]:
        _inpth0 = "_%s_sentiment_data.txt" % split
        _inpth1 = "_%s_tense_data.txt" % split
		_outpth = "_%s_whole_data.txt" % split

        _inpth0 = os.path.join(data_pth, _inpth0)
        _inpth1 = os.path.join(data_pth, _inpth1)
		
		_outpth = os.path.join(res_path, _outpth)

        #Creates output data (unlabelled)
        concat_files(_inpth0, _inpth1, _outpth, False)

                                                                                                                                                                              141,14        79%
def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
                                                                                                                                                                            93,14         40%
                                                                                                                                                   1,14          Top
