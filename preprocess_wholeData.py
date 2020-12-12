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
                    label = line.split("\t")[0]
                    f_out.write("\t"+label+"\t") #tense-label sentiment-label sentence
                f_out.write(line.strip() + "\n")
        with open(pth1, errors='ignore') as f1:
            for line in f1.readlines():
                if with_label:
                    label = line.split("\t")[0]
                    f_out.write(label+"\t\t")
                f_out.write(line.strip() + "\n")

def main(args):
    data_pth = "data/%s" % args.data_name
    res_pth = "results/%s" % args.data_name
    for split in ["train","dev","test"]:
        if args.label=="false":
            _inpth0 = "_%s_sentiment_data.txt" % split
            _inpth1 = "_%s_tense_data.txt" % split
            _outpth = "_%s_whole_data.txt" % split
        else:
            _inpth0 = "%s_sentiment_data.txt" % split
            _inpth1 = "%s_tense_data.txt" % split
            _outpth = "%s_whole_data.txt" % split
        
        _inpth0 = os.path.join(data_pth, _inpth0)
        _inpth1 = os.path.join(data_pth, _inpth1)
        _outpth = os.path.join(res_pth, _outpth)

        if args.label=="false":
            #Creates output data (unlabelled)
            concat_files(_inpth0, _inpth1, _outpth, False)
        else:
            concat_files(_inpth0, _inpth1, _outpth, True)
                                                                                                                                                                              

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--label', type=str, default='false')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
                                                                                                                                                                
                                                                                                                                                  
