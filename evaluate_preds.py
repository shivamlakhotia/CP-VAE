import pandas as pd
import argparse
from utils.bleu import compute_bleu
from utils.text_utils import MonoTextData
import torch
from classifier import CNNClassifier
import os
import numpy as np

def getPreds(model, eval_data):
    preds = []
    for batch_data in eval_data:
        batch_size = batch_data.size(0)
        logits = model(batch_data)
        probs = torch.sigmoid(logits)
        preds.append(np.argmax(probs, axis=1))
    return preds


def main(args):
    print("Entering eval_preds.py...")
    data_pth = "results/%s" % args.data_name
    train_pth = os.path.join(data_pth, "_train_whole_data.txt") #Default vocab is taken from train data
    train_data = MonoTextData(train_pth, False, vocab=100000)
    vocab = train_data.vocab

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_pth = os.path.join(data_pth, args.source_file_name) #Classify the given source file's contents
    print("Classifying data in ",source_pth)
    source_data = MonoTextData(source_pth,False,vocab=100000)  
    source_data_vocab = source_data.vocab
    source_data = source_data.create_data_batch(64,device,batch_first=True)

    target_pth = "results/%s" % args.data_name
    target_pth = os.path.join(target_pth, args.target_file_name) #save the generated output into the target file
	
    source = pd.read_csv(source_pth, sep="\n", header=None)
    source.columns = ["content"]
    #target = pd.read_csv(target_pth, names=['content','sentiment-label','tense-label'], sep='\t')
    target = pd.DataFrame(columns=['content','sentiment-label','tense-label'])
    target.head()


    # Classification 
    for style in ["tense","sentiment"]:
        #model = CNNClassifier(len(vocab), 300, [1,2,3,4,5], 500, 0.5).to(device)
        print("Classifying ",style)
        model_path = "checkpoint/{}-{}-classifier.pt".format(args.data_name,style)
        checkpoint = torch.load(model_path)
        #model = CNNClassifier(len(checkpoint['embedding.weight']), 300, [1,2,3,4,5], 500, 0.5).to(device)
        print(len(checkpoint['embedding.weight']),len(source_data_vocab))
        model = CNNClassifier(len(checkpoint['embedding.weight']), 300, [1,2,3,4,5], 500, 0.5).to(device)
        model.load_state_dict(checkpoint)
        #break
        
        model.eval()
        content = []
        predictions = []
        with torch.no_grad():
            print("Number of batches = ", len(source_data))
            idx = 0
            for batch_data in source_data:
                print("Evaluating batch ",idx)
                logits = model(batch_data)
                probs = torch.sigmoid(logits)
                y_hat = list((probs > 0.5).long().cpu().numpy())
                predictions.extend(y_hat)
                idx = idx + 1
                #break

        label = "{}-label".format(style)
        #print("Number of sentences = ",len(content))
        print("Length of predictions = ",len(predictions))
        #print(predictions)
        target['content'] = source["content"]
       # print("Content:")
       # print(target['content'])
        target[label] = predictions
        #print("Predictions:")
        #print(target[label])
        print("No of sentences = ",len(target))
        print(target.head())
        
    target.to_csv(target_pth,sep='\t')
    print("Output written to ",target_pth)
    
def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--source_file_name', type=str, default='_train_whole_data.txt')
    parser.add_argument('--target_file_name', type=str, default='train_input_data.csv')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
        
    main(args)
