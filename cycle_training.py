#!/usr/bin/env python
# coding: utf-8

# ## Import dependencies


import os
import argparse

import torch
from torch.nn import CrossEntropyLoss
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import re
import numpy as np
import random
import math

from nltk.tokenize import word_tokenize

from datasets import load_dataset
from datasets import load_metric
from torch.utils.data import DataLoader

from torch.optim import AdamW
from transformers import get_scheduler

from tqdm.auto import tqdm

import sys
import json


# ## Optional in-script arguments passing




# sys.argv=['--gpu_device','3',
#           '--num_epochs','1',
#           '--output_dir','/home/ubuntu/playground/',
#           '--data2text_model','/home/ubuntu/experiments/webnlg-t5-triplets2text/base/100',
#           '--text2data_model','/home/ubuntu/experiments/webnlg-t5-text2triplets/base/100',
#           '--text_file','/home/ubuntu/data/webnlg-t5-text2triplets/100/train.source',
#           '--data_file','/home/ubuntu/data/webnlg-t5-triplets2text/100/train.source',
#           '--do_train','--do_eval','--do_test','--do_generate',
#           '--scorer_model','/home/ubuntu/experiments/scorer/synthetic_data/webnlg/scorer-3/',
#           '--data2text_validation_file','/home/ubuntu/data/webnlg-t5-triplets2text/val.tsv',
#           '--text2data_validation_file','/home/ubuntu/data/webnlg-t5-text2triplets/val.tsv',
#           '--data2text_test_file','/home/ubuntu/data/webnlg-t5-triplets2text/test.tsv',
#           '--text2data_test_file','/home/ubuntu/data/webnlg-t5-text2triplets/test.tsv']


# ## Arguments Processing




parser = argparse.ArgumentParser()
## General
parser.add_argument("--config_file", default=None, type=str,
                    help="Optional use of config file for passing the arguments")
parser.add_argument("--output_dir", default=None, type=str,
                    help="The output directory where the model predictions and checkpoints will be written")
parser.add_argument("--gpu_device", default=0, type=int,
                    help="GPU device id")
parser.add_argument("--bertscore_gpu_device", default=0, type=int,
                    help="GPU device id for bertscore model")

## Model
parser.add_argument("--t5_tokenizer", default="t5-base", type=str,
                    help="Tokenizer for T5 models")
parser.add_argument("--data2text_model", default=None, type=str,
                    help="Local or Huggingface transformer's path to the data2text model")
parser.add_argument("--text2data_model", default=None, type=str,
                    help="Local or Huggingface transformer's path to the text2data_model model")
## Data
parser.add_argument("--text_file", default=None, type=str,
                    help="Text used for cycle training (text-data-text cycle)")
parser.add_argument("--data_file", default=None, type=str,
                    help="Data used for cycle training (data-text-data cycle)")

## Generation Parameters
parser.add_argument("--max_input_length", default=256, type=int,
                    help="Maximum input length including prompt after tokenization")
parser.add_argument("--min_output_length", default=3, type=int,
                    help="Minimum output length")
parser.add_argument("--max_output_length", default=256, type=int,
                    help="Maximum output length")
parser.add_argument("--num_beams", default=4, type=int,
                    help="Number of beams for beam search")
parser.add_argument("--no_repeat_ngram_size", default=0, type=int,
                    help="No repeat ngram size")
parser.add_argument("--length_penalty", default=0, type=float,
                    help="Length penalty")

## Cycle Training Parameters
parser.add_argument("--do_train", action='store_true',
                    help="Whether to run training.")
parser.add_argument("--seed", default=1, type=int,
                    help="Random seed")
parser.add_argument("--num_epochs", default=25, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training")
parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for evaluation")
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help="Number of updates steps to accumulate before performing a backward/update pass; effective training batch size equals to per_gpu_train_batch_size * gradient_accumulation_steps")
parser.add_argument("--data2text_learning_rate", default=3e-4, type=float,
                    help="The initial learning rate of AdamW for the data2text model; larger learning rate is suggested for T5 families")
parser.add_argument("--text2data_learning_rate", default=3e-4, type=float,
                    help="The initial learning rate of AdamW for the text2data model; larger learning rate is suggested for T5 families")
parser.add_argument('--scheduler_type', type=str, default="linear",
                    help="Learning rate scheduler type (linear/cosine/cosine_with_restarts/polynomial/constant/constant_with_warmup)")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Scheduler warmup steps")


## Adaptive Cycle Training Parameters
parser.add_argument("--adaptive_type", default=0, type=int,
                    help="0: No adaptive learning; 1: adaptive instance weighted loss; 2: adaptive learning rate")
parser.add_argument("--scorer_model_tokenizer", default="roberta-base", type=str,
                    help="Tokenizer for the scorer model")
parser.add_argument("--scorer_model", default=None, type=str,
                    help="Local path to the scorer model")

## Evaluation Parameters
parser.add_argument("--do_eval", action='store_true',
                    help="Whether to run eval on the dev set")
parser.add_argument("--data2text_validation_file", default=None, type=str,
                    help="The development set of the data2text task")
parser.add_argument("--text2data_validation_file", default=None, type=str,
                    help="The development set of the text2data task")

parser.add_argument("--do_generate", action='store_true',
                    help="Whether to run generation for the evaluation of the dev set")

parser.add_argument("--do_test", action='store_true',
                    help="Whether to run eval on the test set")
parser.add_argument("--data2text_test_file", default=None, type=str,
                    help="The test set of the data2text task")
parser.add_argument("--text2data_test_file", default=None, type=str,
                    help="The test set of the text2data task")

## Model Selection & Saving Strategy
parser.add_argument('--save_epochs', type=int, default=1,
                    help="Save model every X updates epochs")
parser.add_argument('--selection_metric', type=str, default="loss",
                    help="The metric used for model section; --do_generate required for metric other than loss")
parser.add_argument('--delta', type=float, default=0.01,
                    help="Minimum requirement of improvement")
parser.add_argument('--patience', type=int, default=3,
                    help="Terminate the training after n epochs without any improvement")

# args = parser.parse_args(sys.argv)
args = parser.parse_args()



if args.config_file != None:
    with open(args.config_file, 'r') as f:
        args.__dict__ = json.load(f)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print(args.__dict__)
with open(os.path.join(args.output_dir,'config.dict'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)



random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
device = torch.device('cuda:'+str(args.gpu_device)) if torch.cuda.is_available() else torch.device('cpu')


# ## Initialize T5 tokenizer, text2data & data2text model and set up model config




tokenizer = T5Tokenizer.from_pretrained(args.t5_tokenizer)
if args.text2data_model != None:
    model_text2data = T5ForConditionalGeneration.from_pretrained(args.text2data_model)
    model_text2data.config.task_specific_params["summarization"]={
      "early_stopping": True,
      "length_penalty": args.length_penalty,
      "max_length": args.max_output_length,
      "min_length": args.min_output_length,
      "no_repeat_ngram_size": args.no_repeat_ngram_size,
      "num_beams": args.num_beams,
      "prefix": ""
    }
    model_text2data.to(device)


if args.data2text_model != None:
    model_data2text = T5ForConditionalGeneration.from_pretrained(args.data2text_model)
    model_data2text.config.task_specific_params["summarization"]={
      "early_stopping": True,
      "length_penalty": args.length_penalty,
      "max_length": args.max_output_length,
      "min_length": args.min_output_length,
      "no_repeat_ngram_size": args.no_repeat_ngram_size,
      "num_beams": args.num_beams,
      "prefix": ""
    }
    model_data2text.to(device)


# ## Initialize scorer




if args.scorer_model != None:
    tokenizer_scorer = RobertaTokenizer.from_pretrained(args.scorer_model_tokenizer)
    model_scorer = RobertaForSequenceClassification.from_pretrained(args.scorer_model,num_labels=1)
    model_scorer.to(device)


# ## Process training data and initialize training parameters




if args.do_train:
    def tokenize_content(sample):
        return tokenizer(sample['text'],padding = 'max_length', truncation=True, max_length=args.max_input_length)
    
    text = load_dataset('text', data_files=args.text_file)
    tokenized_text = text.map(tokenize_content, batched=True)
    tokenized_text.set_format('torch',['attention_mask','input_ids'],output_all_columns=True)
    text_dataloader = DataLoader(tokenized_text['train'],shuffle=True,batch_size=args.per_gpu_train_batch_size)

    triplets = load_dataset('text', data_files=args.data_file)
    tokenized_triplets = triplets.map(tokenize_content, batched=True)
    tokenized_triplets.set_format('torch',['attention_mask','input_ids'],output_all_columns=True)
    triplets_dataloader = DataLoader(tokenized_triplets['train'],shuffle=True,batch_size=args.per_gpu_train_batch_size)
    
    optimizer_text2data = AdamW(list(model_text2data.parameters()),lr=args.text2data_learning_rate)
    optimizer_data2text = AdamW(list(model_data2text.parameters()),lr=args.data2text_learning_rate)
    
    num_text_training_steps = args.num_epochs * len(text_dataloader)
    num_data_training_steps = args.num_epochs * len(triplets_dataloader)
    
    lr_scheduler_text2data = get_scheduler(
        args.scheduler_type,
        optimizer = optimizer_text2data,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = num_text_training_steps
    )

    lr_scheduler_data2text = get_scheduler(
        args.scheduler_type,
        optimizer = optimizer_data2text,
        num_warmup_steps = args.warmup_steps,
        num_training_steps = num_data_training_steps
    )


# ## Process validation data




def tokenize_val_content(sample):
    return tokenizer(sample['source'],padding = 'max_length', truncation=True, max_length=args.max_input_length)

if args.do_eval:
    if args.text2data_validation_file != None:
        text2triplets_val = load_dataset('csv', data_files={'dev':args.text2data_validation_file},delimiter='\t')
        tokenized_text2triplets_val = text2triplets_val.map(tokenize_val_content, batched=True)
        tokenized_text2triplets_val.set_format('torch',['attention_mask','input_ids'],output_all_columns=True)
        text2triplets_val_dataloader = DataLoader(tokenized_text2triplets_val['dev'],shuffle=False,batch_size=args.per_gpu_eval_batch_size) 
    
    if args.data2text_validation_file != None:
        triplets2text_val = load_dataset('csv', data_files={'dev':args.data2text_validation_file},delimiter='\t')
        tokenized_triplets2text_val = triplets2text_val.map(tokenize_val_content, batched=True)
        tokenized_triplets2text_val.set_format('torch',['attention_mask','input_ids'],output_all_columns=True)
        triplets2text_val_dataloader = DataLoader(tokenized_triplets2text_val['dev'],shuffle=False,batch_size=args.per_gpu_eval_batch_size) 


# ## Process testing data




if args.do_test:    
    if args.text2data_test_file != None:
        text2triplets_test = load_dataset('csv', data_files={'test':args.text2data_test_file},delimiter='\t')
        tokenized_text2triplets_test = text2triplets_test.map(tokenize_val_content, batched=True)
        tokenized_text2triplets_test.set_format('torch',['attention_mask','input_ids'],output_all_columns=True)
        text2triplets_test_dataloader = DataLoader(tokenized_text2triplets_test['test'],shuffle=False,batch_size=args.per_gpu_eval_batch_size) 
    
    if args.data2text_test_file != None:
        triplets2text_test = load_dataset('csv', data_files={'test':args.data2text_test_file},delimiter='\t')
        tokenized_triplets2text_test = triplets2text_test.map(tokenize_val_content, batched=True)
        tokenized_triplets2text_test.set_format('torch',['attention_mask','input_ids'],output_all_columns=True)
        triplets2text_test_dataloader = DataLoader(tokenized_triplets2text_test['test'],shuffle=False,batch_size=args.per_gpu_eval_batch_size) 


# ## Main cycle training function that performs training on one direction




def train_one_direction(model1,model2,data_loader,num_training_steps,optimizer,lr_scheduler,inter_type):
    model1.eval() # freezed model
    model2.train() # trained model
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(range(math.ceil((len(data_loader))/args.gradient_accumulation_steps)))
    
    step = 0
    batch_loss = 0
    total_loss = 0
    batch_score = 0
    total_score = 0
    
    for batch in data_loader:
        # prepare inputs
        raw_input = batch.pop('text')
        model1_input = {k: v.to(device) for k, v in batch.items()}

        # generate intermediate outputs
        with torch.no_grad():
            intermediate_outputs = model1.generate(**model1_input, min_length=args.min_output_length, max_length=args.max_output_length, num_beams=args.num_beams, early_stopping=True)
        
        decoded_intermediate_outputs = tokenizer.batch_decode(intermediate_outputs,skip_special_tokens=True)
        
        # hot prefixing
        scorer_input = None
        if inter_type == 'text':
            decoded_intermediate_outputs = ['Extract Triplets: ' + item for item in decoded_intermediate_outputs]
            scorer_decoded_intermediate_outputs = [im_text+' '+im_triplets.replace('Generate in English','Triplets') for im_text,im_triplets in zip(decoded_intermediate_outputs,raw_input)]
        elif inter_type == 'data':
            decoded_intermediate_outputs = ['Generate in English: ' + item for item in decoded_intermediate_outputs]
            scorer_decoded_intermediate_outputs = [im_text+' '+im_triplets.replace('Generate in English','Triplets') for im_text,im_triplets in zip(raw_input, decoded_intermediate_outputs)]
        
        # prepare labels and inputs
        tokenized_intermediate_outputs = tokenizer(decoded_intermediate_outputs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length)
        model2_label = [t.split(': ')[1] for t in raw_input]
        processed_labels = tokenizer(model2_label, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length)['input_ids']
        processed_labels[processed_labels==0]=-100
        tokenized_intermediate_outputs['labels']=processed_labels
        
        model2_input = {k: v.to(device) for k, v in tokenized_intermediate_outputs.items()}

        # scoring
        if args.scorer_model != None:
            tokenized_scorer_input = tokenizer_scorer(scorer_decoded_intermediate_outputs, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_length)
            scorer_input = {k: v.to(device) for k, v in tokenized_scorer_input.items()}

            with torch.no_grad():
                score = model_scorer(**scorer_input).logits

        # final outputs and loss calculation
        outputs = model2(**model2_input)
        
        loss = None
        logits = outputs.logits
    
        loss_fct = CrossEntropyLoss(ignore_index=-100,reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), model2_input['labels'].view(-1))
        
        # handling of adaptive cycle training
        if args.adaptive_type==1:
            loss = (loss * score).mean()
        else:
            loss = loss.mean()
            if args.scorer_model != None:
                score = torch.mean(score)
        
        if args.adaptive_type==2:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*score

        
        loss = loss / args.gradient_accumulation_steps
        if args.scorer_model != None:
            score = score / args.gradient_accumulation_steps
        
        loss.backward()
        
        if args.scorer_model != None:
            batch_score += score.item()
            total_score += score.item()
        
        batch_loss += loss.item()
        total_loss += loss.item()
    
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.scorer_model != None:
                progress_bar.set_description("Train Batch Loss: %f Score: %f" %(batch_loss,batch_score))
            else:
                progress_bar.set_description("Train Batch Loss: %f" %(batch_loss))
            batch_loss=0
            batch_score=0
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        step += 1
    return total_loss, total_score


# ## Evaluation function




metric_meteor = load_metric("meteor")
metric_bleu = load_metric("bleu")
metric_bertscore = load_metric("bertscore",device=args.bertscore_gpu_device)
print ("bertscore device = ", args.bertscore_gpu_device)
metric_rouge = load_metric("rouge")

def eval_model(model,data_loader,test_mode=False):
    model.eval()
    progress_bar = tqdm(range(math.ceil((len(data_loader))/args.gradient_accumulation_steps)))
    step = 0
    batch_loss = 0
    total_loss = 0
    source_texts=[]
    generated_texts=[]
    target_texts=[]
    for batch in data_loader:
        raw_input = batch.pop('source')
        model_label = batch.pop('target')
        model_input = {k: v.to(device) for k, v in batch.items()}
        
        processed_labels = tokenizer(model_label, return_tensors="pt", padding='max_length', truncation=True, max_length=args.max_input_length)['input_ids']
        processed_labels[processed_labels==0]=-100
        model_input['labels']=processed_labels.to(device)

        with torch.no_grad():
            outputs = model(**model_input)
        
        if args.do_generate:
            del model_input['labels']
            with torch.no_grad():
                generated_outputs = model.generate(**model_input, min_length=args.min_output_length, max_length=args.max_output_length, num_beams=args.num_beams, early_stopping=True)
            decoded_outputs = tokenizer.batch_decode(generated_outputs,skip_special_tokens=True)
            generated_texts += decoded_outputs
            target_texts += model_label
            source_texts += raw_input
        loss = outputs.loss
        total_loss += loss.item()
        
        loss = loss / args.gradient_accumulation_steps
        
        batch_loss += loss.item()
        
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            progress_bar.set_description("Eval Batch Loss: %f" %(batch_loss))
            batch_loss=0
            progress_bar.update(1)
        
        step += 1
    
    resulting_metrics={'loss':total_loss/len(data_loader)}
    if args.do_generate or test_mode:
        meteor_p = []
        meteor_g = []
        bleu_p = []
        bleu_g = []
        bertscore_p = []
        bertscore_g = []
        rouge_p = []
        rouge_g = []
        last_s = ""
        for s, p, g in zip(source_texts,generated_texts,target_texts):
            s = s.strip()
            p = p.strip()
            g = g.replace("<s>","").strip()

            if last_s == s:
                g_tokens = word_tokenize(g)
                meteor_g[-1].append(' '.join(g_tokens))
                bleu_g[-1].append(g_tokens)
                bertscore_g[-1].append(g)
                rouge_g[-1].append(g)
            else:
                p_tokens = word_tokenize(p)
                g_tokens = word_tokenize(g)
                meteor_p.append(' '.join(p_tokens))
                meteor_g.append([' '.join(g_tokens)])
                bleu_p.append(p_tokens)
                bleu_g.append([g_tokens])
                bertscore_p.append(p)
                bertscore_g.append([g])
                rouge_p.append(p)
                rouge_g.append([g])
            last_s = s
        resulting_metrics['meteor']=metric_meteor.compute(predictions = meteor_p, references=meteor_g)['meteor']
        resulting_metrics['bleu']=metric_bleu.compute(predictions = bleu_p, references=bleu_g)['bleu']
        resulting_metrics['bertscore'] = np.mean(metric_bertscore.compute(predictions = bertscore_p, references=bertscore_g, lang='en')['f1'])
        rouge_result = metric_rouge.compute(predictions=rouge_p, references=rouge_g)
        resulting_metrics['rouge1']=rouge_result['rouge1'].mid.fmeasure
        resulting_metrics['rouge2']=rouge_result['rouge2'].mid.fmeasure
        resulting_metrics['rougeL']=rouge_result['rougeL'].mid.fmeasure

    return resulting_metrics, generated_texts


# ## Training and validating scripts




if args.do_train:
    epoch_progress_bar = tqdm(range(args.num_epochs))
    text2data_best = 1000000
    text2data_best_metircs = None
    text2data_patience = 0
    data2text_best = 1000000
    data2text_best_metrics = None
    data2text_patience = 0
    for epoch in range(args.num_epochs):
        epoch_progress_bar.set_description("Cycle/Epoch %d: "%(epoch))
        print("\nTraining: data-text-data direction")
        total_text2data_loss, total_text2data_score  = train_one_direction(model_data2text, model_text2data, triplets_dataloader, num_data_training_steps, optimizer_text2data, lr_scheduler_text2data,'text')
        print('model_text2data - train total loss: %.4f score: %.4f' %(total_text2data_loss,total_text2data_score))

        if args.do_eval and args.text2data_validation_file != None:
            text2data_dev_metrics, _ = eval_model(model_text2data,text2triplets_val_dataloader)
            total_text2data_dev_loss = text2data_dev_metrics['loss']
            if args.selection_metric == 'loss':
                text2data_selection_metric = text2data_dev_metrics[args.selection_metric]
            else:
                text2data_selection_metric = 1 - text2data_dev_metrics[args.selection_metric]
            print('model_text2data - dev total loss: %.4f' %(total_text2data_dev_loss))
            print(text2data_dev_metrics)

            if text2data_patience <= args.patience and text2data_best-text2data_selection_metric>=args.delta:
                text2data_best = text2data_selection_metric
                text2data_best_metircs = text2data_dev_metrics
                text2data_patience=0
                model_text2data.save_pretrained(os.path.join(args.output_dir,'text2data-best'))
                print("text2data-best saved: epoch/cycle %d"%(epoch))
            else:
                text2data_patience+=1

        if epoch % args.save_epochs == 0:
            model_text2data.save_pretrained(os.path.join(args.output_dir,'text2data-' +str(epoch)))
            print('text2data-' +str(epoch)+' saved')
            
        print("\nTraining: text-data-text direction")
        total_data2text_loss, total_data2text_score = train_one_direction(model_text2data, model_data2text, text_dataloader, num_text_training_steps, optimizer_data2text, lr_scheduler_data2text,'data')
        print('model_data2text - train total loss: %.4f score: %.4f' %(total_data2text_loss,total_data2text_score))

        if args.do_eval and args.data2text_validation_file != None:
            data2text_dev_metrics, _ = eval_model(model_data2text,triplets2text_val_dataloader)
            total_data2text_dev_loss = data2text_dev_metrics['loss']
            if args.selection_metric == 'loss':
                data2text_selection_metirc = data2text_dev_metrics[args.selection_metric]
            else:
                data2text_selection_metirc = 1 - data2text_dev_metrics[args.selection_metric]
            print('model_data2text - dev total loss: %.4f' %(total_data2text_dev_loss))
            print(data2text_dev_metrics)

            if data2text_patience <= args.patience and data2text_best-data2text_selection_metirc>=args.delta:
                data2text_best = data2text_selection_metirc
                data2text_best_metircs = data2text_dev_metrics
                data2text_patience=0
                model_data2text.save_pretrained(os.path.join(args.output_dir,'data2text-best'))
                print("data2text-best saved: epoch/cycle %d"%(epoch))
            else:
                data2text_patience+=1

        if epoch % args.save_epochs == 0:
            model_data2text.save_pretrained(os.path.join(args.output_dir,'data2text-' +str(epoch)))
            print('data2text-' +str(epoch)+' saved')
                
        epoch_progress_bar.update(1)
        if text2data_patience>args.patience and data2text_patience>args.patience:
            print("Both models exceed the patience, training terminated")
            break
    print("\nTraining completed")
    if args.do_eval:
        print("\nBest data2text model:")
        print(data2text_best_metircs)
        del model_data2text
        
        print("\nBest text2data model:")
        print(text2data_best_metircs)
        del model_text2data
        
    if args.do_test:
        if args.data2text_test_file != None:
            model_data2text = T5ForConditionalGeneration.from_pretrained(os.path.join(args.output_dir,'data2text-best'))
            model_data2text.to(device)
        
        if args.text2data_test_file != None:
            model_text2data = T5ForConditionalGeneration.from_pretrained(os.path.join(args.output_dir,'text2data-best'))
            model_text2data.to(device)
        


# ## Testing scripts


if args.do_test:
    if args.data2text_model != None and args.data2text_test_file != None:    
        data2text_test_metrics, data2text_generations = eval_model(model_data2text,triplets2text_test_dataloader,test_mode=True)
        print("\ndata2text test:")
        print(data2text_test_metrics)
        out = open(os.path.join(args.output_dir,'data2text.generations'),'w')
        out.write('\n'.join(data2text_generations))
        out.close()
        
    if args.text2data_model != None and args.text2data_test_file != None:    
        text2data_test_metrics, text2data_generations = eval_model(model_text2data,text2triplets_test_dataloader,test_mode=True)
        print("\ntext2data test:")
        print(text2data_test_metrics)
        out = open(os.path.join(args.output_dir,'text2data.generations'),'w')
        out.write('\n'.join(text2data_generations))
        out.close()

