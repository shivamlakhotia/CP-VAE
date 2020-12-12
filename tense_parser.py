from nltk import word_tokenize, pos_tag
import argparse
import operator
import os
import pandas as pd
import nltk
import difflib
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def determine_tense_input(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)

    tense = {}
    tense["future"] = len([word for word in tagged if word[1] == "MD"])
    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) 
    result = max(tense.items(), key=operator.itemgetter(1))[0]
    if result=='past':
        return 0
    else:
        return 1

def main(args):
    data_pth = "results/%s" % args.data_name
    target_pth = "results/%s" % args.data_name
    input_pth = os.path.join(data_pth, args.input)
    output_pth = os.path.join(target_pth,args.output)
    source = pd.read_csv(input_pth, sep="\t")
    target = pd.DataFrame(columns=['content','tense-label','sentiment-label'])
    target.head()
    tenseLabels = []
    i = 0
    identical = 0
    classifier_labels = source['tense-label']
    content = []
    sentiment_labels = []
    output_tense_labels = []
    for sentence in source['content']:
        print('Finding tense of sentence ',i)
        parser_label = determine_tense_input(sentence)
        classifier_label = classifier_labels.get(i)
        tenseLabels.append(parser_label)
        if parser_label == classifier_label:
            identical = identical + 1
            content.append(sentence)
            sentiment_labels.append(source['sentiment-label'].get(i))
            output_tense_labels.append(parser_label)
        i = i + 1

    target['content'] = content#source['content']
    target['tense-label'] = output_tense_labels#tenseLabels#source['tense-label']
    target['sentiment-label'] = sentiment_labels
    target.to_csv(output_pth,sep='\t',index=False)
    print("Output written to ",target_pth)
    
    print("Total no of sentences = ", len(source['content']))
    print("No of identical labels = ", identical)
    print("Identical ratio = ", identical/len(source['content']))
    print("Identical percentage = ",((identical/len(source['content']))*100))
    #sm=difflib.SequenceMatcher(None,target['classifier-tense-label'],target['parser-tense-label'])
    #print("Similarity between two tense labels = ",sm.ratio())
    print("Total no of new identical sentences writtent = ",len(content))

def add_args(parser):
    parser.add_argument('--data_name', type=str, default='yelp')
    parser.add_argument('--input', type=str, default='train_new_data.csv')
    parser.add_argument('--output', type=str, default='train_identical_data.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    main(args)

