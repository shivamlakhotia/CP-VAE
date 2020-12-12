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
        probs = torch.sigmoid(logits) # Tensor([0.1, 0.3, 0.7, 0.9])
        preds.append(np.argmax(probs, axis=1))
    return preds


def main(args):
    print("Entering eval_preds.py...")
    data_pth = "data/%s" % args.data_name
    temp = "_train_%s_data.txt" % args.style
    train_pth = os.path.join(data_pth, temp) #Default vocab is taken from train data
    train_data = MonoTextData(train_pth, False, vocab=100000)
    vocab = train_data.vocab

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_pth = os.path.join(data_pth, args.source_file_name) #Classify the given source file's contents
    print("Classifying data in ",source_pth)
    source_data = MonoTextData(source_pth,True,vocab=100000)  
    source_data_vocab = source_data.vocab
    source_data = source_data.create_data_batch(64,device,batch_first=True)

    target_pth = "results/%s" % args.data_name
    target_pth = os.path.join(target_pth, args.target_file_name) #save the generated output into the target file
	
    source = pd.read_csv(source_pth, sep="\t", header=None)
    source.columns = ["label","content"]
    #target = pd.read_csv(target_pth, names=['content','sentiment-label','tense-label'], sep='\t')
    target = pd.DataFrame(columns=['content','sentiment-label','tense-label'])
    target.head()

    # Classification 
    if args.style=="sentiment":
        #model = CNNClassifier(len(vocab), 300, [1,2,3,4,5], 500, 0.5).to(device)
        print("Classifying tense on given sentiment labeled data")
        model_path = "checkpoint/{}-{}-classifier.pt".format(args.data_name,"tense")
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
                probs = torch.sigmoid(logits) #prob(1)
                # y_hat = list((probs > 0.5).long().cpu().numpy()) 
                # predictions.extend(y_hat)
                #retaining probability values itself so that we can threshold later and remove less confident sentences
                predictions.extend(list(probs.cpu().numpy())) 
                idx = idx + 1
                #break

        label = "{}-label".format("tense")
        #print("Number of sentences = ",len(content))
        print("Length of predictions = ",len(predictions))
        #print(predictions)
       # print("Content:")
       # print(target['content'])
        final_content = []
        final_sentiment_label = []
        final_tense_label = []
        i = 0
        for pred in predictions:
            pred_1 = pred #prob(1) 0.3 0.8 
            pred_0 = 1-pred_1 #prob(0) 0.7 0.2
            if pred_1 >= args.confidence or pred_0 >= args.confidence: #model is 80% confidently predicting at least one label, so retain the sentence
                if pred_1>=args.confidence: 
                    final_tense_label.append(1)
                else:
                    final_tense_label.append(0)
                final_content.append(source["content"].get(i))
                final_sentiment_label.append(source["label"].get(i))
            i = i + 1

        target['content'] = final_content#source["content"]        
        target[label] = final_tense_label#predictions
        #print("Predictions:")
        #print(target[label])
        target['sentiment-label'] = final_sentiment_label#source["label"]
        print("No of sentences, after retaining only 80% confident predictions = ",len(target))
        print(target.head())
    else:
        print("Classifying sentiment on tense labeled data")
        model_path = "checkpoint/{}-{}-classifier.pt".format(args.data_name,"sentiment")
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
                # y_hat = list((probs > 0.5).long().cpu().numpy()) 
                # predictions.extend(y_hat)
                #retaining probability values itself so that we can threshold later and remove less confident sentences
                predictions.extend(list(probs.float().cpu().numpy())) 
                idx = idx + 1
                #break

        label = "{}-label".format("sentiment")
        #print("Number of sentences = ",len(content))
        print("Length of predictions = ",len(predictions))

        final_content = []
        final_sentiment_label = []
        final_tense_label = []
        i = 0
        for pred in predictions:
            pred_1 = pred #prob(1) 0.3 0.8 
            pred_0 = 1-pred_1 #prob(0) 0.7 0.2
            if pred_1 >= args.confidence or pred_0 >= args.confidence: #model is 80% confidently predicting at least one label, so retain the sentence
                if pred_1>=args.confidence: 
                    final_sentiment_label.append(1)
                else:
                    final_sentiment_label.append(0)
                final_content.append(source["content"].get(i))
                final_tense_label.append(source["label"].get(i))
            i = i + 1

        #print(predictions)
        target['content'] = final_content#source["content"]
       # print("Content:")
       # print(target['content'])
        target[label] = final_sentiment_label#predictions
        #print("Predictions:")
        #print(target[label])
        target['tense-label'] = final_tense_label#source["label"]
        print("No of sentences, after retaining only 80% confident predictions = ",len(target))
        print(target.head())
        
    target.to_csv(target_pth,sep='\t')
    print("Output written to ",target_pth)
    
def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--source_file_name', type=str, default='_train_whole_data.txt')
    parser.add_argument('--target_file_name', type=str, default='train_input_data.csv')
    parser.add_argument('--style', type=str, default="sentiment")
    parser.add_argument('--confidence',type=float,default=0.8)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
        
    main(args)
