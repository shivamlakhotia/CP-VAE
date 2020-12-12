import os
import pandas as pd
import config
import argparse

def concat_files(pth0, pth1, outpth):
    # a = pd.read_csv(pth0)
    # b = pd.read_csv(pth1)
    # b = b.dropna(axis=1)
    # merged = a.merge(b, on='title')
    # merged.to_csv(outpth, index=False)

    #combine all files in the list
    all_filenames = [pth0,pth1]
    combined_csv = pd.concat([pd.read_csv(f,sep="\t",index_col=0) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv(outpth, index=False, encoding='utf-8-sig',sep="\t")


def main(args):
    data_pth = "results/%s" % args.data_name
    res_pth = "results/%s" % args.data_name
    
    _inpth0 = os.path.join(data_pth, args.file1)
    _inpth1 = os.path.join(data_pth, args.file2)
    _outpth = os.path.join(res_pth,args.output)
    
    concat_files(_inpth0, _inpth1, _outpth)
                                                                                                                                                                              

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--file1', type=str, default='train_labeled_data.csv')
    parser.add_argument('--file2', type=str, default='train_labeled_data_1.csv')
    parser.add_argument('--output', type=str, default='train_new_data.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
