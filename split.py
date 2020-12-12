import os
import pandas as pd
import config
import argparse

def split_files(input_pth,outpth1,outpth2):
    source = pd.read_csv(input_pth, sep="\t")
    sentiment_labels = source['sentiment-label']
    tense_labels = source['tense-label']
    content = source['content']
    with open(outpth1, "w") as f_out:
        i = 0
        for label in sentiment_labels:
            f_out.write(str(label)+"\t")
            line = content.get(i)
            f_out.write(line.strip() + "\n")
            i = i + 1
    f_out.close()
    with open(outpth2, "w") as f_out:
        i = 0
        for label in tense_labels:
            f_out.write(str(label)+"\t")
            line = content.get(i)
            f_out.write(line.strip() + "\n")
            i = i + 1
    f_out.close()

def main(args):
    data_pth = "results/%s" % args.data_name
    res_pth = "results/%s" % args.data_name
    
    _inpth = os.path.join(data_pth, args.input)
    _outpth1 = os.path.join(res_pth,args.output1)
    _outpth2 = os.path.join(res_pth,args.output2)
    
    split_files(_inpth, _outpth1, _outpth2)
                                                                                                                                                                              

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--input', type=str, default='train_identical_data.csv')
    parser.add_argument('--output1', type=str, default='train_identical_sentiment.txt')
    parser.add_argument('--output2', type=str, default='train_identical_tense.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)
