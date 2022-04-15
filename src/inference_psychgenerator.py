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
import pickle 
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
import string
import time
import math

from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from gpt2_wrapper import GPT2_WRAPPER

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)}

"""====================== METHODS DEFINITIONS ======================"""

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def truncating_padding_sentence(tokens, block_size):
    if (len(tokens) > block_size):
        original_tokens_len = block_size
        tokens = tokens[:block_size]
    else:
        original_tokens_len = len(tokens)
        tokens = tokens + ["<|pad|> "]*(block_size - len(tokens))
    return tokens, original_tokens_len    


def create_attention_mask(sentence_length, seq_length, gpt2_config, mask_type):
    # args:
    #   sentence_length is length of real text, from <|sos|>  to <|endoftext|>
    #   seq_length is length with <|pad|> (32, 64, 128, ...)
    
    if mask_type == "encoder_mask":
        print("Please set mask_type as: decoder_mask")
        return 
    if mask_type == "decoder_mask":
        # attention, the triangular matrix is [seq_length,seq_length+1] becasuse the original has one more "past" token
        mask_one_head = np.tril(np.ones([seq_length,seq_length+1]),1)
        mask_all_heads = [mask_one_head] * gpt2_config.n_head   
        mask_all_heads = np.array(mask_all_heads)
    return mask_all_heads            
            

def convert_tensor_inference(model, sentence_embedding, args, device, current_generated_sentences):
    

    # convert to tensor and put on device
    sentence_embedding_converted = torch.FloatTensor(sentence_embedding).unsqueeze(0).to(device)
    
    ## DEBUGGING
    # logger.info("sentence_embedding_converted: " + str(sentence_embedding_converted.shape))
    
    # generate sentence
    generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = sentence_embedding_converted, args = args, device = device)
    generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
    first_endoftext = generated_sample.find("<|endoftext|>") 
    generated_sample = str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)])
    count = 1
    while ((generated_sample in current_generated_sentences) and (count<10)):
        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = sentence_embedding_converted, args = args, device = device)
        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
        first_endoftext = generated_sample.find("<|endoftext|>") 
        generated_sample = str(generated_sample[:(first_endoftext)])
        count += 1
                
    current_generated_sentences.append(generated_sample)   
    
    ## DEBUGGING
    # logger.info("len(current_generated_sentences): " + str(len(current_generated_sentences)))    
    # logger.info("current_generated_sentences: " + str(current_generated_sentences))
    # logger.info("generated_sample: " + str(generated_sample))
    # print("count+: "+ str(count))

    # print generated sentence sample 
    generated_sample = generated_sample[6: ]
    print(str(generated_sample))
    
    return current_generated_sentences


def inference_methods(model, args, device):
    
    ### get training data
    data_df = pd.read_csv(args.train_data_file, header = 0, index_col = 0)
    print(data_df.head())


    if args.method == "variables_inference":
        # analyze training data
        sentences_embeddings = data_df.iloc[:,1:].values
        sentences_text = data_df.iloc[:,:1]   
        messages_big5_training = sentences_embeddings
        dimensions_name = data_df.columns[1:]
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # set parameters
        explore_std_range = [-args.k_value,args.k_value+0.01]
        std_step_interval = args.k_value
        std_random_level = 0.00001


        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("\n")
            print("=====")
            print("HIDDEN DIMENISON {} ({}) : ".format(str(i), dimensions_name[i]))   

            ##### generating texts
            print("***** generating text in interval: ")
            # explore the pole
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
                print("\n")
                print("samples around mean + {}*std:".format(round(std_position,2)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    message_big5 = np.copy(means)
                    message_big5[i] = message_big5[i] + std_position*stds[i] + epsilon*stds[i]

                    # concat (user_big5 - not applicable) and message_big5
                    embedding_sample = message_big5

                    # transform to tensor
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(prompting_text = args.prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
                        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                        generated_count += 1
                        first_endoftext = generated_sample.find("<|endoftext|>") 
                        generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                        if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                            generated_samples.append(generated_sample_clean)
                            break
                        
                    # print generated sentence sample
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    print("generated_sample: " + str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]))


    if args.method == "variables_demographics_inference":
        # generate texts with input is message-level big5 score and DEMOGRAPHICS score
        # for each personalities, travel along the dimension of that personality to generate text with different DEMOGRAPHICS score
        # e.g.: generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
        # with different demographics values of [M,F] -> [1,-1], age -> [35 yo]

        # analyze training data
        demographics_variable_name = args.demographics_variable
        dimensions_name = [item for item in data_df.columns[1:] if item!=demographics_variable_name]
        sentences_embeddings = data_df[dimensions_name].values
        messages_big5_training = sentences_embeddings
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)
        sentences_demographics = data_df[demographics_variable_name].values
        demographics_means = sentences_demographics.mean()
        demographics_stds = sentences_demographics.std()

        # set parameters
        explore_std_range = [-args.k_value,args.k_value+0.01]
        std_step_interval = args.generate_interval
        std_random_level = 0.00001


        # set parameters
        explore_std_range = [-args.k_value,args.k_value+0.01]
        std_step_interval = args.k_value
        demographics_explore_std_range = [-args.k_value_demographics,args.k_value_demographics+0.01]
        demographics_std_step_interval = args.k_value_demographics
        std_random_level = 0.00001


        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("\n")
            print("=====")
            print("HIDDEN DIMENISON {} ({}) : ".format(str(i), dimensions_name[i]))   

            ##### generating texts
            # explore the pole
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
                print("\n\n")
                print("=============================================")
                print("====> Samples around mean + {}*std:".format(round(std_position,2)))

                # explore the demographics variables
                for demographic_std_position in np.arange(demographics_explore_std_range[0], demographics_explore_std_range[1], demographics_std_step_interval):  
                    demographic_value = demographics_means + demographics_stds*demographic_std_position
                    print("\n")
                    print("=> Demographics variable [{}] - position {}.std [{}]: ".format(demographics_variable_name, str(demographic_std_position), str(round(demographic_value,2))))   

                    print("generation avoid repeating!")
                    generated_samples = []   # avoid repeated generated_sample
                    for _ in range(args.generate_num):     
                        
                        # sample embedding around embedding + std_position*stds[i]
                        epsilon = np.random.uniform(-std_random_level,std_random_level)
                        message_big5 = np.copy(means)
                        message_big5[i] = message_big5[i] + std_position*stds[i] + epsilon*stds[i]

                        # concat message_big5 and demographics variables
                        embedding_sample = list(message_big5) + [demographic_value]
                        embedding_sample = np.array(embedding_sample)

                        # transform to tensor
                        embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
            
                        # generate sentence
                        generated_count = 0    
                        while True:
                            generated_sample, decoder_attentions_sample = model.inference(prompting_text = args.prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
                            generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                            generated_count += 1
                            first_endoftext = generated_sample.find("<|endoftext|>") 
                            generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                            if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                                generated_samples.append(generated_sample_clean)
                                break
                            
                        # print generated sentence sample
                        first_endoftext = generated_sample.find("<|endoftext|>") 
                        print("generated_sample: " + str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]))

"""====================== MAIN FUNCTION ======================"""

# main function
def main():
    parser = argparse.ArgumentParser()

    # dataset/save path parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")                      
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--input_file", default=None, type=str, required=False,
                        help="Where to read input file if needed.")
    parser.add_argument("--output_file", default=None, type=str, required=False,
                        help="Where to save output file if needed.")
        
    # model parameters
    parser.add_argument("--gpt2_model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--gpt2_model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--latent_size", default=-1, type=int, required=True,
                        help="Size of latent VAE layer.")    
    parser.add_argument('--demographics_variable', type=str, default=None, required=False,
                        help="The name of demographics variable if there is.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 
    
    # training parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
        
    # generating parameters
    parser.add_argument("--inference_test", default=1, type=int)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--generate_num", type=int, default=None)
    parser.add_argument("--generate_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompting_text", type=str, default="I like to")
    parser.add_argument("--k_value", type=float, default=3.0)
    parser.add_argument("--k_value_demographics", type=float, default=3.0)
    parser.add_argument("--generate_interval", type=float, default=3.0)
    parser.add_argument("--evaluating_embedding", type=str, default=None)
    parser.add_argument("--evaluating_text", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)

    # other generating parameters
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # other parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")    
    parser.add_argument("--from_checkpoint", action='store_true',
                        help="To initialize model or load from a checkpoint.")     
    # parsing parameters
    args = parser.parse_args()
    
    
    # =========== checking parameters and setting up  =========== #

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


    # =========== bulilding model and inferencing  =========== #
    # Building model
    gpt2_config_class, gpt2_class, tokenizer_class = MODEL_CLASSES[args.gpt2_model_type]
    gpt2_config = gpt2_config_class.from_pretrained(args.gpt2_model_name_or_path, cache_dir = None)
    latent_size = args.latent_size
    model = GPT2_WRAPPER(gpt2_config, latent_size)
    
    # Load from checkpoint model
    model.from_pretrained(args)
    if args.block_size <= 0:  # modify args.block_size variable
        args.block_size = model.tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, model.tokenizer.max_len_single_sentence)
    
    # Send model to GPU
    model.to(args.device)    

    # Logging info
    logger.info("Inference parameters %s", args)
    

    # Testing inference
    args.gpt2_config = model.gpt2_config
    inference_methods(model, args, device)    
    
if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time()

    # total time
    print("Total running time: {} seconds ({} hours)".format(str(round(end_time-start_time)), str(round((end_time-start_time)/3600,2))))      




