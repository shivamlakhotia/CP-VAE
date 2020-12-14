import pandas as pd
import argparse
import os

def main(args):
    data_pth = "data/%s" % args.data_name
    target_pth = "results/%s" % args.data_name
    source_pth = os.path.join(data_pth, args.input)
    output_pthA = os.path.join(target_pth,args.outputA)
    output_pthB = os.path.join(target_pth,args.outputB)
    sentiment = args.sentiment

    source = pd.read_csv(source_pth, sep="\t") #sent-A\tsent-B
    #target = pd.read_csv(target_pth, names=['content','sentiment-label','tense-label'], sep='\t')
    targetA = pd.DataFrame(columns=['content','tense-label','sentiment-label'])
    targetB = pd.DataFrame(columns=['content','tense-label','sentiment-label'])
    targetA.head()
    targetB.head()

    contentA = []
    contentB = []
    for i in range(len(source)):
        print("Parsing sentence ",i)
        sentA = source.values[i][0]
        sentB = source.values[i][1]
        contentA.append(sentA)
        contentB.append(sentB)

    targetA['content'] = contentA
    targetA['tense-label'] = [-1] * len(source)
    targetA['sentiment-label'] = [args.sentiment] * len(source)
    targetA.to_csv(output_pthA,sep='\t',index=False)
    print("Output written to ",output_pthA)

    targetB['content'] = contentB
    targetB['tense-label'] = [-1] * len(source)
    targetB['sentiment-label'] = [1-args.sentiment] * len(source)
    targetB.to_csv(output_pthB,sep='\t',index=False)
    print("Output written to ",output_pthB)

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--input', type=str, default='reference.0')
    parser.add_argument('--outputA', type=str, default='eval_A0.csv')
    parser.add_argument('--outputB', type=str, default='eval_B1.csv')
    parser.add_argument('--sentiment',type=int, default=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)

