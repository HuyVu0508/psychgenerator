# import libraries
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import numpy as np
import torch
import pandas as pd


from torch.utils.data import DataLoader, Dataset, RandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (AdamW, get_linear_schedule_with_warmup,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)


from gpt2_wrapper import GPT2_WRAPPER
print("from gpt2_wrapper import GPT2_WRAPPER")

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}


"""====================== METHODS DEFINITIONS ======================"""

# defining method for creating mask
def create_attention_mask(sentence_length, seq_length, gpt2_config, mask_type):
    # args:
    #   sentence_length is length of real text, from <|sos|> to <|endoftext|>
    #   seq_length is length with <|pad|> (32, 64, 128, ...)
    
    if mask_type == "encoder_mask":
        print("Please set mask_type as: decoder_mask")
        return 
    if mask_type == "decoder_mask":
        # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
        mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1)
        mask_all_heads = [mask_one_head] * gpt2_config.n_head   
        mask_all_heads = np.array(mask_all_heads)

        # # CHANGED!
        # # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
        # mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1) 
        # mask_one_head[:,0] = 0 # ignore the conditioned vector
        # mask_all_heads = [mask_one_head] * gpt2_config.n_head   
        # mask_all_heads = np.array(mask_all_heads)   

    return mask_all_heads            

def truncating_padding_sentence(tokens, block_size):
    if (len(tokens) > block_size):
        original_tokens_len = block_size
        tokens = tokens[:block_size]
    else:
        original_tokens_len = len(tokens)
        tokens = tokens + ["<|pad|>"]*(block_size - len(tokens))
    return tokens, original_tokens_len    

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, args):
        
        # reading data file
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(args.block_size) + '_' + filename)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)       
                
            
            # reading file
            self.examples = []

            data_df = pd.read_csv(file_path, header = 0, index_col = False)
            for i, record in data_df.iterrows(): 

                # read data
                sentence_id = record["message_id"]
                sentence_text = str(record["message"])
                sentence_embedding = np.array(record[2:].values).astype(float)

                # tokenize sentence
                sentence_tokenized = tokenizer.tokenize(sentence_text)

                # decoder_input
                decoder_input = ["<|sos|>"] + sentence_tokenized
                decoder_input, decoder_input_len = truncating_padding_sentence(decoder_input, args.block_size)
                decoder_input = tokenizer.convert_tokens_to_ids(decoder_input)
                decoder_input = np.array(decoder_input)
                # decoder_output
                decoder_label = sentence_tokenized + ["<|endoftext|>"]
                decoder_label, decoder_label_len = truncating_padding_sentence(decoder_label, args.block_size)
                decoder_label = tokenizer.convert_tokens_to_ids(decoder_label)
                decoder_label = np.array(decoder_label)

                # decoder_attention_mask
                # decoder_attention_mask = create_attention_mask(decoder_input_len, args.block_size, args.gpt2_config, "decoder_mask")
                decoder_attention_mask = 0

                # append to examples list
                training_sentence = dict({"sentence_embedding": sentence_embedding, "sentence_text": sentence_text, "decoder_input": decoder_input, "decoder_attention_mask": decoder_attention_mask, "decoder_label": decoder_label})  
                self.examples.append(training_sentence)


            # print examples of training set
            for i in range(5):
                example = self.examples[i]
                logger.info("decoder_input: " + str(example["decoder_input"]))
                logger.info("decoder_label: " + str(example["decoder_label"]))           
                logger.info("decoder_input: " + str(tokenizer.decode(example["decoder_input"].tolist(), clean_up_tokenization_spaces=True))) 
                logger.info("decoder_label: " + str(tokenizer.decode(example["decoder_label"].tolist(), clean_up_tokenization_spaces=True)))

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
    
def load_and_cache_examples(args, file_path, tokenizer):
    dataset = TextDataset(tokenizer, file_path=file_path, args=args)
    return dataset    

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)   

def loss_fn(decoder_lm_logits, target, ignore_index):
    
    # Negative Log Likelihood
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
    # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
    NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  

    return NLL_loss

def loss_perplexity_fn(decoder_lm_logits, target, ignore_index):
    
    # Negative Log Likelihood
    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
    # # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
    # print("decoder_lm_logits: " + str(decoder_lm_logits.shape))
    # NLL_loss = loss_fct(decoder_lm_logits, target)
    # print("NLL_loss: " + str(NLL_loss.shape))  # expect [batch_size of 1 GPU, 1]
    # perplexity = torch.exp(NLL_loss)
    # print("perplexity: " + str(perplexity.shape)) # expect [batch_size of 1 GPU, 1]

    NLL_loss_batch = []
    perplexity_batch = []
    for i in range(decoder_lm_logits.shape[0]):
        NLL_loss_onesample = loss_fct(decoder_lm_logits[i], target[i])
        perplexity_onesample = torch.exp(NLL_loss_onesample)
        NLL_loss_batch.append(NLL_loss_onesample)
        perplexity_batch.append(perplexity_onesample)

    return NLL_loss_batch, perplexity_batch    

"""====================== TRAIN/EVALUATE FUNCTION ======================"""

# train and evaluate function
def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """

    # ===== Setting up
    # summary writer
    tb_writer = SummaryWriter()
    
    print("DEBUGGING!")
    print("train_dataset: " + str(len(train_dataset)))
    print(train_dataset[0])
    print("train_batch_size: " + str(args.per_gpu_train_batch_size * max(1, args.n_gpu)))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},  
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


    if args.from_checkpoint:
        global_step = args.start_step
        t_total += args.start_step
    else:
        global_step = 0


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)    


    # ===== Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * 1)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    if args.from_checkpoint:
        logger.info("  Starting from checkpoint {}, total optimization steps = {}".format(args.start_step, t_total))
    else:
        logger.info("  Total optimization steps = %d", t_total)




    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)


    # create decoder_attention_mask here instead of in the loop to save running time 
    decoder_attention_mask_onesample = create_attention_mask(0, args.block_size, args.gpt2_config, "decoder_mask")
    decoder_attention_mask_batchsize = torch.tensor([decoder_attention_mask_onesample] * args.train_batch_size)

    loss_report = []
    eval_loss_report = []
    eval_perplexity_loss_report = []
    eval_current_step = []
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
 
            sentence_embedding = batch["sentence_embedding"].float()
            decoder_input = batch["decoder_input"].long() 
            decoder_label = batch["decoder_label"].long()        
            if len(batch["decoder_attention_mask"]) == args.train_batch_size:
                decoder_attention_mask = decoder_attention_mask_batchsize.long() 
            else:
                decoder_attention_mask = torch.tensor([decoder_attention_mask_onesample] * len(batch["decoder_attention_mask"])).long()             
            sentence_embedding = sentence_embedding.to(args.device)
            decoder_input = decoder_input.to(args.device)
            decoder_label = decoder_label.to(args.device)
            decoder_attention_mask = decoder_attention_mask.to(args.device)


            # forward pass (change and edit with VAE code)
            decoder_lm_logits = model(sentence_embedding, decoder_input, decoder_attention_mask, args.device)


            # compute loss  
            NLL_loss = loss_fn(decoder_lm_logits, decoder_label, tokenizer.convert_tokens_to_ids(["<|pad|>"])[0])    
            ## DEBUGGING
            loss = NLL_loss


            # # DEBUGGING
            # print("sentence_embedding: " + str(sentence_embedding))
            # input_text = tokenizer.decode(decoder_input[0].tolist(), clean_up_tokenization_spaces=True)
            # print("input_text: " + input_text)
            # predictions = torch.nn.functional.softmax(decoder_lm_logits, dim = -1)
            # logger.info("decoder_input: " + str(tokenizer.decode(decoder_input[0].tolist(), clean_up_tokenization_spaces=True))) 
            # logger.info("decoder_label: " + str(tokenizer.decode(decoder_label[0].tolist(), clean_up_tokenization_spaces=True)))
            # prediction_text = tokenizer.decode(torch.argmax(predictions[0], dim=-1).tolist(), clean_up_tokenization_spaces=True)
            # first_endoftext = prediction_text.find("<|endoftext|>") 
            # logger.info("predictions: " + str(prediction_text[:(first_endoftext + 13) if first_endoftext>0 else len(prediction_text)])) 


            # process loss across GPUs, batches then backwards
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps


            loss_report.append(loss.data.cpu().numpy())

            # run loss backward
            loss.backward()


            # accummulte enough step, step backward
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                # print loss
                if args.printing_steps > 0 and global_step % args.printing_steps == 0:


                    # if global_step % (10*args.printing_steps) == 0:
                    #     # DEBUGGING
                    #     predictions = torch.nn.functional.softmax(decoder_lm_logits, dim = -1)
                    #     logger.info("decoder_input: " + str(tokenizer.decode(decoder_input[0].tolist(), clean_up_tokenization_spaces=True))) 
                    #     logger.info("decoder_label: " + str(tokenizer.decode(decoder_label[0].tolist(), clean_up_tokenization_spaces=True)))
                    #     prediction_text = tokenizer.decode(torch.argmax(predictions[0], dim=-1).tolist(), clean_up_tokenization_spaces=True)
                    #     first_endoftext = prediction_text.find("<|endoftext|>") 
                    #     logger.info("predictions: " + str(prediction_text[:(first_endoftext + 13) if first_endoftext>0 else len(prediction_text)])) 


                    # Log metrics
                    print("Current training step: " + str(global_step))
                    print("Average current training loss of the latests {} steps: {}".format(str(args.printing_steps), str(np.mean(loss_report[-args.printing_steps:]))))  

                    

                # logging 
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss


                # save checkpoints
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # evaluate
                    if args.evaluate_during_training:
                        # set model to eval
                        model.eval()

                        # running train function
                        eval_loss, eval_perplexity = evaluate(args, eval_dataset, model, tokenizer)
                        eval_loss_report.append(eval_loss)
                        eval_perplexity_loss_report.append(eval_perplexity)
                        eval_current_step.append(global_step)

                        # set model to train
                        model.train()

                    # save model
                    loss_reports = {"loss_report":loss_report, "eval_loss_report":eval_loss_report, "eval_perplexity_loss_report":eval_perplexity_loss_report, "eval_current_step":eval_current_step}
                    if args.from_checkpoint:    # concatenate with results from checkpoint if training from check point
                        loss_reports_from_checkpoint = pickle.load(open(args.output_dir + "/checkpoint-{}".format(str(args.start_step)) + "/loss_reports.pkl", "rb"))
                        for key in loss_reports.keys():
                            loss_reports[key] = loss_reports_from_checkpoint[key] + loss_reports[key]
                    model.module.save_pretrained(args, output_dir, loss_reports)                    
                    logger.info("Saving model checkpoint to %s", output_dir)
                    _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # save final loss_reports
    loss_reports = {"loss_report":loss_report, "eval_loss_report":eval_loss_report, "eval_perplexity_loss_report":eval_perplexity_loss_report, "eval_current_step":eval_current_step}
    if args.from_checkpoint:    # concatenate with results from checkpoint if training from check point
        loss_reports_from_checkpoint = pickle.load(open(args.output_dir + "/checkpoint-{}".format(str(args.start_step)) + "/loss_reports.pkl", "rb"))
        for key in loss_reports.keys():
            loss_reports[key] = loss_reports_from_checkpoint[key] + loss_reports[key]

    # close summary writer
    tb_writer.close()

    return global_step, tr_loss, loss_reports

def evaluate(args, eval_dataset, model, tokenizer):

    # ===== Setting up
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # ===== Evaluate!
    print("\n")
    print("************************************")
    print("***** Start running evaluating *****")
    print("  Num examples = " + str(len(eval_dataset)))
    print("  Instantaneous batch size per GPU = " + str(args.per_gpu_eval_batch_size))

    # create decoder_attention_mask
    decoder_attention_mask_onesample = create_attention_mask(0, args.block_size, args.gpt2_config, "decoder_mask")

    loss_report = []
    perplexity_report = []
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=False)):

        sentence_embedding = batch["sentence_embedding"].float()
        decoder_input = batch["decoder_input"].long() 
        decoder_label = batch["decoder_label"].long()        
        decoder_attention_mask = torch.tensor([decoder_attention_mask_onesample] * len(batch["decoder_attention_mask"])).long() 
        sentence_embedding = sentence_embedding.to(args.device)
        decoder_input = decoder_input.to(args.device)
        decoder_label = decoder_label.to(args.device)
        decoder_attention_mask = decoder_attention_mask.to(args.device)

        with torch.no_grad():
            model.eval()


            # forward pass (change and edit with VAE code)
            decoder_lm_logits = model(sentence_embedding, decoder_input, decoder_attention_mask, args.device)


            # compute loss  
            loss, perplexity = loss_perplexity_fn(decoder_lm_logits, decoder_label, tokenizer.convert_tokens_to_ids(["<|pad|>"])[0])    


            # # DEBUGGING
            # predictions = torch.nn.functional.softmax(decoder_lm_logits, dim = -1)
            # logger.info("decoder_input: " + str(tokenizer.decode(decoder_input[0].tolist(), clean_up_tokenization_spaces=True))) 
            # logger.info("decoder_label: " + str(tokenizer.decode(decoder_label[0].tolist(), clean_up_tokenization_spaces=True)))
            # prediction_text = tokenizer.decode(torch.argmax(predictions[0], dim=-1).tolist(), clean_up_tokenization_spaces=True)
            # first_endoftext = prediction_text.find("<|endoftext|>") 
            # logger.info("predictions: " + str(prediction_text[:(first_endoftext + 13) if first_endoftext>0 else len(prediction_text)])) 


            # process loss across GPUs, batches then backwards
            if args.n_gpu > 1:
                loss = torch.stack(loss)    # concatenate across all GPUs
                # print("aggregated NLL_loss: " + str(loss.shape))    # expect [batchsize of 3 GPUs, 1]
                perplexity = torch.stack(perplexity)    # concatenate across all GPUs
                # print("aggregated perplexity: " + str(perplexity.shape))    # expect [batchsize of 3 GPUs, 1]
            elif args.n_gpu == 1:
                loss = torch.tensor(loss)
                perplexity = torch.tensor(perplexity)


            loss_report.extend(loss.data.cpu().numpy())
            perplexity_report.extend(perplexity.data.cpu().numpy())

    # report results
    print("=== Evaluating results ===")
    print("Average evaluating loss: " + str(np.mean(loss_report)))
    print("Average perplexity: " + str(round(np.mean(perplexity_report),3)))
    # print("loss_report: " + str(np.array(loss_report).shape))
    # print("perplexity: " + str(np.array(perplexity_report).shape))
    print("***********************************")
    print("***** End running evaluating *****")
    print("\n")

    return np.mean(loss_report), np.mean(perplexity_report)


"""====================== MAIN FUNCTION ======================"""


# main function
def main():
    
    # =========== parameters parsing =========== #
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--from_checkpoint", action='store_true',
                        help="To initialize model or load from a checkpoint.")    
    parser.add_argument("--start_step", type=int,
                        help="The checkpoint number.")    
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    
    # model parameters
    parser.add_argument("--gpt2_model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--gpt2_model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 

    # training parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
    parser.add_argument("--frozen_layers", type=str, default='None', 
                        help="Layers to be frozen while training.")
   

    # other training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--printing_steps', type=int, default=50,
                        help="Print every X updates steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    
    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #
    # checking parameters
    if args.eval_data_file is None and args.do_eval :
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file or remove the --do_eval argument.")
    if args.from_checkpoint is None and args.do_eval :
        raise ValueError("Cannot do evaluation without specified checkpoint.")
    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir and not(args.from_checkpoint):
            raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    
    # setting things up    
    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")    # CHECK! make sure we use all 3 GPUs
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    # Set seed
    set_seed(args)


    # =========== bulilding model and training/evaluating  =========== #

    # Building model
    gpt2_config_class, gpt2_class, tokenizer_class = MODEL_CLASSES[args.gpt2_model_type]
    gpt2_config = gpt2_config_class.from_pretrained(args.gpt2_model_name_or_path, cache_dir = None)
    latent_size = args.latent_size
    model = GPT2_WRAPPER(gpt2_config, latent_size)

    
    # Initialize / Load from checkpoint model
    if args.do_train:
        if args.from_checkpoint == False:
            model.initialize_model(args)    # initialize model with pretrained GPT2
        else:
            model.from_pretrained(args)
    elif args.do_eval:
        model.from_pretrained(args)
    if args.block_size <= 0:  # modify args.block_size variable
        args.block_size = model.tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, model.tokenizer.max_len_single_sentence)


    # count number of params
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    print("Number of parameters: " + str(pytorch_total_params))
    


    # Send model to GPU
    model.to(args.device)

    ### Set parallel running
    ## use DataParallel
    model = torch.nn.DataParallel(model) 
    ## use DistributedDataParallel
    # import torch.distributed as dist
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '6006'
    # print("here1")
    # dist.init_process_group("nccl",rank=0, world_size=args.n_gpu)
    # print("here2")
    # model = torch.nn.parallel.DistributedDataParallel(model) 


    # Logging info
    logger.info("Inference parameters %s", args)


    # Training
    if args.do_train:
        
        # DEBUGGING
        # print model parameters 
        logger.info("TRANSFORM_MATRIX")    
        for name, param in model.module.transform_matrix.named_parameters():
            logger.info(name + ' - ' + str(param.requires_grad))
        logger.info("DECODER")     
        for name, param in model.module.decoder.named_parameters():
            logger.info(name + ' - ' + str(param.requires_grad))
         
            
        #  freeze layers
        if args.frozen_layers is not None:
            frozen_layers = args.frozen_layers.split(" ")
            for name, param in model.named_parameters():
                if any(".{}.".format(str(frozen_layer)) in name for frozen_layer in frozen_layers):
                    logger.info("frozen params: " + name)
                    param.requires_grad = False
            
            
        # load train_dataset
        args.gpt2_config = model.module.gpt2_config
        train_dataset = load_and_cache_examples(args, args.train_data_file, model.module.tokenizer)
        if args.evaluate_during_training:
            eval_dataset = load_and_cache_examples(args, args.eval_data_file, model.module.tokenizer)
        else:
            eval_dataset = None

        # set model to train
        model.train()

        # running train function
        global_step, tr_loss, loss_reports = train(args, train_dataset, eval_dataset, model, model.module.tokenizer)
        logger.info(" global_step = %s", global_step)

        # saving model
        model.module.save_pretrained(args, args.output_dir, loss_reports)
        
        # good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))        

   
    # Evaluating
    if args.do_eval:

        # load train_dataset
        args.gpt2_config = model.module.gpt2_config
        eval_dataset = load_and_cache_examples(args, args.eval_data_file, model.module.tokenizer)

        # set model to eval
        model.eval()

        # running train function
        evaluate(args, eval_dataset, model, model.module.tokenizer)

if __name__ == "__main__":
    main()        



