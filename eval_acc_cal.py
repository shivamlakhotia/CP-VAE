import pandas as pd
import argparse
from utils.text_utils import MonoTextData
import torch
from classifier import CNNClassifier, evaluate
import os

data_pth = "data"
file_path = os.path.join(data_pth, "acc_M3_eval_A1.csv")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_pth = os.path.join("data", "yelp/train_identical_sentiment_90_tense.csv")
train_data = MonoTextData(train_pth, True, vocab=100000)
vocab = train_data.vocab

eval_data = MonoTextData(file_path, True,vocab = vocab)

# Classification Accuracy

eval_data, eval_sent_label, eval_tense_label = eval_data.create_data_batch_labels(64, device, batch_first=True)

model_sent = CNNClassifier(len(vocab), 300, [1, 2, 3, 4, 5], 500, 0.5).to(device)
model_sent.load_state_dict(torch.load("checkpoint/ours-yelp/yelp-sentiment-classifier.pt"))
model_sent.eval()
acc_sent = 100 * evaluate(model_sent, eval_data, eval_sent_label)
print("Sent Acc: %.2f" % acc_sent)

