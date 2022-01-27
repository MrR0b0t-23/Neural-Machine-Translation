import torch 
import torch.nn as nn
from torchtext.legacy.data import Field, BucketIterator, TabularDataset
import numpy as np
import pandas as pd
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


import sys, os, random
from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize

torch.backends.cudnn.enabled = False

INDIC_NLP_LIB_HOME=r"C:\Users\admin\Desktop\Machine Translation\indic_nlp_library-master"
INDIC_NLP_RESOURCES=r"C:\Users\admin\Desktop\Machine Translation\indic_nlp_resources-master"
sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
common.set_resources_path(INDIC_NLP_RESOURCES)
loader.load()

from torchtext.data import get_tokenizer
tokenize_eng = get_tokenizer("toktok")

def tokenize_tamil(text):
    return [token for token in indic_tokenize.trivial_tokenize(text)]

tamil = Field(tokenize=tokenize_tamil, lower=True, use_vocab = True, sequential = True,
               init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng,lower=True, use_vocab = True, sequential = True,
               init_token="<sos>", eos_token="<eos>")

fields = {'tamil': ('t', tamil), 'english': ('e', english)}

train_data, val_data, test_data = TabularDataset.splits(path ='Dataset',train = 'tamil-eng-train.csv',
                                                        test = 'tamil-eng-test.csv',
                                                        validation = 'tamil-eng-val.csv',
                                                        format = 'csv', fields = fields)

tamil.build_vocab(train_data, max_size=10000, min_freq = 2)
print(f"Unique tokens in source (ta) vocabulary: {len(tamil.vocab)}")

english.build_vocab(train_data, max_size=10000, min_freq =2)
print(f"Unique tokens in source (en) vocabulary: {len(english.vocab)}")

num_epochs = 20
learning_rate = 0.001
batch_size = 64

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
input_size_encoder = len(tamil.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
embedding_size = 300
hidden_size = 1024
num_layers = 2
dropout = 0.5

step = 0

PATH = "Checkpoints/seq2seq.pt"

print("DEVICE :",device)

train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, val_data, test_data), 
                                                                      batch_size = batch_size, 
                                                                      sort_within_batch=True,
                                                                      sort_key=lambda x: len(x.t),
                                                                      device = device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        
    def forward(self, x):
        # x-shape ==> (seq_len, batch_size)
        embedding = self.dropout(self.embedding(x))
        #embedding shape ==> (seq_len, batch_size, embedding_size)
        output, (hidden, cell) = self.lstm(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, dropout = p)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        #x-shape ==> (1, batch_size)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        #embedding shape ==> (1, batch_size, embedding_size)
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        #outputs shape ==> (1, batch_size, hidden_size)
        prediction = self.fc(outputs)
        #prediction shape ==> (1, batch_size, len_of_english_vocab)
        prediction = prediction.squeeze(0)
        
        return prediction, hidden, cell

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, eng_vocab, device):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.eng_vocab = eng_vocab
        self.device = device
        
    def forward(self, x, y, tfr = 0.5):
        
        batch_size = x.shape[1]
        y_len = y.shape[0]
        y_vocab_size = self.eng_vocab
        
        outputs = torch.zeros(y_len, batch_size, y_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(x)
        x = y[0] #start token
        for t in range(1, y_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            # output shape ==> (batch_size, english_vocab_len)
            outputs[t] = output
            #best_guess = output.detach().numpy().argmax(1)
            best_guess = output.argmax(1)
            x= y[t] if random.random() < tfr else best_guess
        return outputs

encoder_lstm = Encoder(input_size_encoder, embedding_size, hidden_size, num_layers, dropout).to(device)
decoder_lstm = Decoder(input_size_decoder, embedding_size, hidden_size, output_size, 
                       num_layers, dropout).to(device)
model = seq2seq(encoder_lstm, decoder_lstm, input_size_decoder, device).to(device)

print(model)

class NMT(pl.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model = model      

    def training_step(self, batch, batch_idx):
        output, y, loss = self._shared_step_(batch, batch_idx)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        output, y, loss = self._shared_step_(batch, batch_idx)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        output, y, loss = self._shared_step_(batch, batch_idx)
        self.log('test_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def _shared_step_(self, batch, batch_idx):
        x = batch.t.to(device)
        y = batch.e.to(device)
        output = self.model(x, y)
        output = output[1:].reshape(-1, output.shape[2])
        y = y[1:].reshape(-1)
        loss = criterion(output, y)
        return output, y, loss

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")
        
# default logger used by trainer
logger = TensorBoardLogger(save_dir=os.getcwd(), version=2, name="seq2seq")
checkpoint = ModelCheckpoint(dirpath='Checkpoints/')

NMT_model = NMT(model).to(device)
callbacks = [PrintCallback(), checkpoint]

trainer = Trainer(gradient_clip_algorithm= 'norm', gpus= 1, max_epochs = 25,
                  num_nodes= -1, logger = logger, auto_select_gpus= True, auto_lr_find=True,
                  callbacks=callbacks, enable_checkpointing=True, weights_save_path='/',
                  log_every_n_steps = 64)

trainer.fit(NMT_model, train_iterator, valid_iterator)