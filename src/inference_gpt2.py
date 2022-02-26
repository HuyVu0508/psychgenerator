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


def inference_test_1(model, args, device):
    
    ### Get embeddings
    data_df = pd.read_csv(args.train_data_file, header = 0, index_col = 0)
    sentences_embeddings = data_df.iloc[:,1:].values
    sentences_text = data_df.iloc[:,:1]   



    ### Analyze embeddings
    if args.method == "method_1":
        # method 1 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)
        
        # set parameters
        std_steps = 5
        std_step_interval = 0.1
        std_random_level = 0
        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            # explore the positive direction
            for step in range(0, std_steps + 1, 1):
                std_position = (step+1)*std_step_interval
                
                print("samples around mean + {}*std:".format(std_position))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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
        
            # explore the negative direction
            for step in range(0, std_steps + 1, 1):
                std_position = (step+1)*std_step_interval
                
                print("samples around mean - {}*std:".format(std_position))
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding - std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] - std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
              
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_2":
        
        # transform data_df to df
        df = data_df.copy()
        df.columns = ["message_id", "text"] + [str(latent_dim) for latent_dim in range(len(sentences_embeddings[0]))]
        df.index = df["message_id"]
        print(df.columns)
        print(df.index[:5])
        

        # compute *std score
        hidden_size = len(sentences_embeddings[0])
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)    
        for i in range(hidden_size):
            df['*std_' + str(i)] = round((df[str(i)] - means[i])/stds[i],2)
            
            
        ## get unique sentence with highest score for each hidden dimension
        print("(=)(=)(=) unique sentence highest score (=)(=)(=)")
        for i in range(hidden_size):        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            top_count = 15
            print("(*) top max score: ")
            sorted_df = df.sort_values(by=[str(i)], ascending=False)
            printed_sentences = []
            for index, row in sorted_df.iterrows():
                if row['text'] not in printed_sentences:
                    count = len(sorted_df[sorted_df['text'] == row['text']])
                    print("{} - score: {} - count: {}".format(row['text'], round(row[str(i)],2), count))
                    printed_sentences.append(row['text'])
                if len(printed_sentences)== top_count:
                    break
                
            print("(*) top min score: ")
            sorted_df = df.sort_values(by=[str(i)], ascending=True)
            printed_sentences = []
            for index, row in sorted_df.iterrows():
                if row['text'] not in printed_sentences:
                    count = len(sorted_df[sorted_df['text'] == row['text']])
                    print("{} - *std score: {} - count: {}".format(row['text'], round(row['*std_' + str(i)],2), count))
                    printed_sentences.append(row['text'])
                if len(printed_sentences)== top_count:
                    break


    if args.method == "method_3":

        # transform data_df to df
        df = data_df.copy()
        df.columns = ["message_id", "text"] + [str(latent_dim) for latent_dim in range(len(sentences_embeddings[0]))]
        df.index = df["message_id"]
        print(df.columns)
        print(df.index[:5])
        

        # compute *std score
        hidden_size = len(sentences_embeddings[0])
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)    
        for i in range(hidden_size):
            df['*std_' + str(i)] = round((df[str(i)] - means[i])/stds[i],2)
        
        
        ## get sentence with highest frequency for each hidden dimension   
        print("(=)(=)(=) sentence highest frequency (=)(=)(=)")
        for i in range(hidden_size):        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            top_count = 8
            std_threshold = 1
            print("(*) top highest frequency above {}*std ".format(str(std_threshold)))
            df_pass_threshold = df.loc[df[str(i)] >= (means[i] + std_threshold*stds[i])]
            df_pass_threshold['text_std'] = df_pass_threshold['text'] + '    ' + df_pass_threshold[('*std_' + str(i))].astype(str)
            print(df_pass_threshold['text_std'].value_counts()[:top_count])
            print("(*) top highest frequency below -{}*std: ".format(str(std_threshold)))
            df_pass_threshold = df.loc[df[str(i)] <= (means[i] - std_threshold*stds[i])]
            df_pass_threshold['text_std'] = df_pass_threshold['text'] + '    ' + df_pass_threshold[('*std_' + str(i))].astype(str)
            print(df_pass_threshold['text_std'].value_counts()[:top_count])
            
            df_pass_threshold.groupby(['text', '*std_' + str(i)])    
   

    if args.method == "method_4":
        # method 4 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)
        
        # set parameters
        explore_std_range = [2.0,4.0]
        std_step_interval = 0.1
        std_random_level = 0
        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            # explore the EXTREME positive direction
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
                print("samples around mean + {}*std:".format(std_position))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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
        

    if args.method == "method_5":
        # method 5 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)
        

        # method for extracting most frequent words
        def most_frequent_words(sentences):
            # tokenize
            sentences = [sentence.split(" ") for sentence in sentences]
            # remove "the world is"
            sentences = [sentence[3:] for sentence in sentences]
            # merge all to one list
            all_words = []
            for sentence in sentences:
                all_words.extend(sentence)
            # remove stop words, remove marks (.,?|)
            processed_all_words = []
            for word in all_words:
                if word not in stopwords.words() and word not in string.punctuation:
                    word = re.sub('['+string.punctuation+']', '', word)
                    if word.strip()!="":
                        processed_all_words.append(word)
            all_words = processed_all_words
            # count frequencies
            df = pd.DataFrame(all_words, columns = ['word'])
            df_counts = df['word'].value_counts()
            return df_counts


        # set parameters
        explore_std_range = [2.0,4.0]

        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            
            # explore the EXTREME positive direction
            value_range = [means[i] + explore_std_range[0]*stds[i], means[i] + explore_std_range[1]*stds[i]]
            value_range_data_df = data_df.loc[(data_df[str(i)] >= value_range[0]) & (data_df[str(i)] <= value_range[1])]
            print("sentences in value range total: " + str(len(value_range_data_df['message'])))
            
            # print words frequencies
            print("words highest frequencies: ")
            df_counts = most_frequent_words(value_range_data_df['message'])
            print(df_counts[:20])


    if args.method == "method_6":
        # method 6 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # method for extracting most frequent words
        def most_frequent_words(sentences):
            # tokenize
            sentences = [sentence.split(" ") for sentence in sentences]
            # remove "the world is"
            sentences = [sentence[3:] for sentence in sentences]
            # merge all to one list
            all_words = []
            for sentence in sentences:
                all_words.extend(sentence)
            # remove stop words, remove marks (.,?|)
            processed_all_words = []
            for word in all_words:
                if word not in stopwords.words() and word not in string.punctuation:
                    word = re.sub('['+string.punctuation+']', '', word)
                    if word.strip()!="":
                        processed_all_words.append(word)
            all_words = processed_all_words
            # count frequencies
            df = pd.DataFrame(all_words, columns = ['word'])
            df_counts = df['word'].value_counts()
            return df_counts


        # set parameters
        explore_std_range = [2.0,4.0]
        stable_mean_range = [-1000,1.5]
        std_step_interval = 0.2
        std_random_level = 0
        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")   
            

            ##### look for high frequent 1-gram
            print("***** words highest frequencies in interval: ")
            # explore the EXTREME positive direction
            interested_dimension_value_range = [means[i] + explore_std_range[0]*stds[i], means[i] + explore_std_range[1]*stds[i]]
            not_interested_dimension_value_range = [ [means[j] + stable_mean_range[0]*stds[j], means[j] + stable_mean_range[1]*stds[j]] for j in range(hidden_size)]           
            value_range_data_df = data_df.loc[(data_df[str(i)] >= interested_dimension_value_range[0]) & (data_df[str(i)] <= interested_dimension_value_range[1])]
            for j in range(hidden_size):
                if j!=i:
                    value_range_data_df = value_range_data_df.loc[(value_range_data_df[str(j)] >= not_interested_dimension_value_range[j][0]) & (value_range_data_df[str(j)] <= not_interested_dimension_value_range[j][1])]

            print("sentences in value range total: " + str(len(value_range_data_df['message'])))
            
            # print words frequencies
            df_counts = most_frequent_words(value_range_data_df['message'])
            print(df_counts[:10])            


            # ##### generating texts
            # print("***** generating text in interval: ")
            # # explore the EXTREME positive direction
            # for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
            
            #     print("samples around mean + {}*std:".format(round(std_position,1)))
            #     print("generation avoid repeating!")
            #     generated_samples = []   # avoid repeated generated_sample
            #     for _ in range(args.generate_num):     
                    
            #         # sample embedding around embedding + std_position*stds[i]
            #         epsilon = np.random.uniform(-std_random_level,std_random_level)
            #         embedding_sample = np.copy(means)
            #         embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
            #         embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
            #         # generate sentence
            #         generated_count = 0    
            #         while True:
            #             generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
            #             generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
            #             generated_count += 1
            #             first_endoftext = generated_sample.find("<|endoftext|>") 
            #             generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
            #             if (generated_sample_clean not in generated_samples) or generated_count >= 10:
            #                 generated_samples.append(generated_sample_clean)
            #                 break
                        
            #         # print generated sentence sample
            #         first_endoftext = generated_sample.find("<|endoftext|>") 
            #         print("generated_sample: " + str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]))


    if args.method == "method_7":
        # method 7 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)

        # set parameters
        explore_std_range = [-1.0,1.0]

        # loop through each dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")  
        
        
            # count percentages in interval
            value_range = [means[i] + explore_std_range[0]*stds[i], means[i] + explore_std_range[1]*stds[i]]
            value_range_data_df = data_df.loc[(data_df[str(i)] >= value_range[0]) & (data_df[str(i)] <= value_range[1])]
            print("sentences in value range total: " + str(len(value_range_data_df['message'])))
            print("percentages: " + str(round(len(value_range_data_df)/len(data_df),2)))


            # test if normal
            from scipy.stats import shapiro
            dimension_data = data_df[str(i)].values
            print(dimension_data[:10])
            print(len(dimension_data))
            stat, p = shapiro(dimension_data)
            alpha = 0.05
            if p > alpha:
                print('Sample looks Gaussian (fail to reject H0)')
            else:
                print('Sample does not look Gaussian (reject H0)')


    if args.method == "method_8":
        # method 8 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # set parameters
        explore_std_range = [2.0,4.0]
        stable_mean_range = [-1000,1.5]
        std_step_interval = 0.2
        std_random_level = 0
        
        # generate sentence for each hidden dimension
        df_texts_generated = []
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")      


            ##### generating texts
            df_texts_generated_onedimension = []
            print("***** generating text in interval: ")
            # explore the EXTREME positive direction
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
            
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
                        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                        generated_count += 1
                        first_endoftext = generated_sample.find("<|endoftext|>") 
                        generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                        if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                            generated_samples.append(generated_sample_clean)
                            break
                        
                    # print generated sentence sample
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    cleaned_sentence = str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)])
                    print("generated_sample: " + cleaned_sentence)


                    # add to list of one dimension
                    df_texts_generated_onedimension.append(cleaned_sentence)

            # add to list of dimensions
            df_texts_generated.append(df_texts_generated_onedimension)

        # save to file
        index = ['dimension_' + str(i) for i in range(hidden_size)]
        columns = ['generated_' + str(i+1) for i in range(len(df_texts_generated[0]))]
        df_texts_generated = pd.DataFrame(df_texts_generated, index = index, columns = columns)
        print(df_texts_generated.head())
        df_texts_generated.to_csv(args.output_dir + "/" + "results_method8.csv", index = True, header = True)


    if args.method == "method_9":
        # method 9 (just use the embedding)
        # analyzing, find std and mean of each hidden dimension    
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # percentages of values belonging to interval [mean + 2.std, mean + 4.std]
        percentages = []
        for i in range(len(means)):
            print("dimension {}:".format(str(i)))
            dimension_range = [means[i]+2.0*stds[i], means[i]+4*stds[i]]
            dimension_count = [ 1 if (item>=dimension_range[0] and item<=dimension_range[1]) else 0 for item in sentences_embeddings[:,i] ]
            dimension_percentage = np.sum(dimension_count)/len(sentences_embeddings[:,i])
            print(dimension_percentage)
            percentages.append(dimension_percentage)
        print("average percentages: " + str(np.mean(percentages)))


        # method for extracting most frequent words
        def most_frequent_words(sentences):
            # tokenize
            sentences = [sentence.split(" ") for sentence in sentences]
            # remove "the world is"
            sentences = [sentence[3:] for sentence in sentences]
            # merge all to one list
            all_words = []
            for sentence in sentences:
                all_words.extend(sentence)
            # remove stop words, remove marks (.,?|)
            processed_all_words = []
            for word in all_words:
                if word not in stopwords.words() and word not in string.punctuation:
                    word = re.sub('['+string.punctuation+']', '', word)
                    if word.strip()!="":
                        processed_all_words.append(word)
            all_words = processed_all_words
            # count frequencies
            df = pd.DataFrame(all_words, columns = ['word'])
            df_counts = df['word'].value_counts()
            return df_counts


        # set parameters
        explore_std_range = [0.5,5.5]
        stable_mean_range = [-1000,1.5]
        std_step_interval = 0.5
        std_random_level = 0


        # generate sentence for each hidden dimension
        df_texts_generated = []
        df_texts_embeddings_csv = []
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("=====")
            print("HIDDEN DIMENISON " + str(i) + ":")      


            ##### generating texts
            df_texts_generated_onedimension = []
            df_texts_embeddings_csv_onedimension = []
            print("***** generating text in interval: ")
            # explore the EXTREME positive direction
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
            
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    torch_embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = torch_embedding_sample, args = args, device = device)
                        generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                        generated_count += 1
                        first_endoftext = generated_sample.find("<|endoftext|>") 
                        generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                        if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                            generated_samples.append(generated_sample_clean)
                            break
                        
                    # print generated sentence sample
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    cleaned_sentence = str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)])
                    cleaned_sentence = cleaned_sentence.replace("<|sos|>","").replace("<|endoftext|>", "")
                    print("generated_sample: <|sos|> " + cleaned_sentence + " <|endoftext|>")


                    # add to list of one dimension
                    df_texts_generated_onedimension.append(cleaned_sentence)
                    df_texts_embeddings_csv_onedimension.append([cleaned_sentence, embedding_sample])

            # add to list of dimensions
            df_texts_generated.append(df_texts_generated_onedimension)
            df_texts_embeddings_csv.extend(df_texts_embeddings_csv_onedimension)


        # ### save to file
        # # texts generated 
        # index = ['dimension_' + str(i) for i in range(hidden_size)]
        # columns = ['generated_' + str(i+1) for i in range(len(df_texts_generated[0]))]
        # df_texts_generated = pd.DataFrame(df_texts_generated, index = index, columns = columns)
        # df_texts_generated.to_csv(args.output_dir + "/" + "results_method9_texts.csv", index = True, header = True)
        # # texts and embeddings csv
        # df_texts_embeddings_csv = [[item[0]] + item[1].tolist() for item in df_texts_embeddings_csv]
        # df_texts_embeddings_csv = pd.DataFrame(df_texts_embeddings_csv)
        # print(df_texts_embeddings_csv)
        # df_texts_embeddings_csv.insert(loc = 0, column = 'message_id', value = ['message_' + str(i) for i in range(len(df_texts_embeddings_csv))])
        # df_texts_embeddings_csv.columns = ['message_id'] + ['message'] + ['dimension_' + str(i) for i in range(hidden_size)]
        # df_texts_embeddings_csv.to_csv(args.output_dir + "/" + "results_method9_texts_embeddings.csv", index = False, header = True)



        # # count words frequencies 
        # n = 10
        # df_words_generated = pd.DataFrame(np.zeros([len(df_texts_generated), n]), index = df_texts_generated.index)
        # for i in range(len(df_texts_generated)):
        #     words_counts = list(most_frequent_words(df_texts_generated.iloc[i,:]).index)
        #     words_counts = words_counts[:10]
        #     df_words_generated.iloc[i,:len(words_counts)] = words_counts

        # # save to file
        # print(df_words_generated.head())
        # df_words_generated.to_csv(args.output_dir + "/" + "results_method9_words.csv", index = True, header = True)


    if args.method == "method_10":
        # generate texts with input is user-level big5 score
        # for each personalities, travel along the dimension of that personality to generate text
        # e.g.: generate text for user-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]

        # method 10 
        # travel along the personality poles ("ope",  "con", "ext", "agr", "neu")   
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # set parameters
        explore_std_range = [-4.0,4.2]
        std_step_interval = 4
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
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_11":
        # generate texts with input is user-level big5 score
        # we look at the real big5 scores from real users from csv file, then generate texts from them
        # e.g.: generate text for user-level [1.2,0.3,-0.3,1.2,1.6], [0.5,0.7,-0.4,-1.2,1.2]

        # method 11
        # get big5 scores from real users
        big5_users_df = pd.read_csv(args.input_file, header=0, index_col=None)
        big5_users_df['user_id'] = big5_users_df['user_id'].astype(str)
        big5_users = []
        for index, row in big5_users_df.iterrows():
            user_big5 = {}
            user_big5['user_id'] = row['user_id']
            # MUST KEEP THIS ORDER! VERY IMPORTANT!!!
            user_big5['big5_scores'] = [row['ope'], row['con'], row['ext'], row['agr'], row['neu']]
            big5_users.append(user_big5)



        # loop through each user, generate texts for that user
        generated_texts_big5_users = []
        for i in tqdm(range(len(big5_users))):
            
            generated_samples = []
            for _ in range(args.generate_num):     
                
                # sample embedding around embedding + std_position*stds[i]
                std_random_level = 0
                epsilon = np.random.uniform(-std_random_level,std_random_level, 5)
                embedding_sample = big5_users[i]['big5_scores']
                embedding_sample = embedding_sample + epsilon
                embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
        
                # generate sentence
                generated_count = 0    
                while True:
                    generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
                    generated_sample =  model.tokenizer.decode(generated_sample[0].tolist(), clean_up_tokenization_spaces=True)
                    generated_count += 1
                    first_endoftext = generated_sample.find("<|endoftext|>") 
                    generated_sample_clean = generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]
                    # remove <|sos|> and <|endoftext|>
                    generated_sample_clean = generated_sample_clean.replace("<|sos|>","").replace("<|endoftext|>","")
                    if (generated_sample_clean not in generated_samples) or generated_count >= 10:
                        generated_samples.append(generated_sample_clean)
                        break
                    
                
                # save to variable
                user_id = big5_users[i]['user_id']
                message_number = str(len(generated_samples))
                message_id = user_id + '_' + message_number
                generated_texts_big5_users.append([user_id, message_id, generated_sample_clean])


                # print generated sentence sample
                first_endoftext = generated_sample.find("<|endoftext|>") 
                print("generated_sample: " + str(generated_sample[:(first_endoftext + 13) if first_endoftext>0 else len(generated_sample)]))


            # write to file every 5 users
            if (i+1)%5==0:
                generated_texts_big5_users_df = pd.DataFrame(generated_texts_big5_users)
                generated_texts_big5_users_df.columns = ['user_id', 'message_id', 'message']
                generated_texts_big5_users_df.to_csv(args.output_file, header=True, index=False)

        # write to file
        generated_texts_big5_users_df = pd.DataFrame(generated_texts_big5_users)
        generated_texts_big5_users_df.columns = ['user_id', 'message_id', 'message']
        generated_texts_big5_users_df.to_csv(args.output_file, header=True, index=False)


    if args.method == "method_12.1":
        # generate texts with input is user-level + message-level big5 score
        # for one person with a fixed user-level big5, travel along the dimension of each personality of message-level big5 to generate text
        # e.g.: with a fixed user-level big5, generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
        
        # method 12.1 
        # user_big5 = [0.22,0.47,-1.30,-1.05,-1.05]
        user_big5 = [0.0,0.0,0.0,0.0,0.0]
        messages_big5_training = sentences_embeddings[:,5:]
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # set parameters
        explore_std_range = [-4.0,4.2]
        std_step_interval = 1
        std_random_level = 0.00001
        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("\n")
            print("=====")
            print("HIDDEN DIMENISON {} ({}) : ".format(str(i), dimensions_name[i]))   
            print("User big5: " + str(user_big5))

            ##### generating texts
            print("***** generating text in interval: ")
            # explore the pole
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
                print("\n")
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    message_big5 = np.copy(means)
                    message_big5[i] = message_big5[i] + std_position*stds[i] + epsilon*stds[i]

                    # concat user_big5 and message_big5
                    embedding_sample = np.array(user_big5 + list(message_big5))

                    # transform to tensor
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_12.2":
        # generate texts with input is user-level + message-level big5 score
        # for one message with a fixed message-level big5, travel along the dimension of each personality of user-level big5 to generate text
        # e.g.: with a fixed message-level big5, generate text for user-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
        
        # method 12.2 
        # message_big5 = [0.22,0.47,-1.30,-1.05,-1.05]
        message_big5 = [0.0,0.0,0.0,0.0,0.0]
        user_big5_training = sentences_embeddings[:,:5]
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        means = np.mean(user_big5_training, axis = 0)
        stds = np.std(user_big5_training, axis = 0)


        # set parameters
        explore_std_range = [-4.0,4.2]
        std_step_interval = 1
        std_random_level = 0.00001
        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            print("\n")
            print("=====")
            print("HIDDEN DIMENISON {} ({}) : ".format(str(i), dimensions_name[i]))   
            print("message big5: " + str(message_big5))

            ##### generating texts
            print("***** generating text in interval: ")
            # explore the pole
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
            
                print("\n")
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    user_big5 = np.copy(means)
                    user_big5[i] = user_big5[i] + std_position*stds[i] + epsilon*stds[i]

                    # concat user_big5 and message_big5
                    embedding_sample = np.array(list(user_big5) + message_big5) 

                    # transform to tensor
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_13":
        # generate texts with input is message-level big5 score
        # for each personalities, travel along the dimension of that personality to generate text
        # e.g.: generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
        
        # method 13
        messages_big5_training = sentences_embeddings
        if "big5" in args.train_data_file:
            dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        elif "depanganx" in args.train_data_file:
            dimensions_name = ["depression",  "anxiety"]
        elif "life" in args.train_data_file or "swl" in args.train_data_file:
            dimensions_name = ["swl"]
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # # set parameters
        # explore_std_range = [-4.0,4.2]
        # std_step_interval = 0.5
        # std_random_level = 0.00001
        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
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
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_13_demographics":
        # generate texts with input is message-level big5 score and DEMOGRAPHICS score
        # for each personalities, travel along the dimension of that personality to generate text with different DEMOGRAPHICS score
        # e.g.: generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
        # with different demographics values of [M,F] -> [1,-1], age -> [35 yo]

        # get sentences embeddings and not demographic variables
        sentences_embeddings = data_df.iloc[:,1:-1].values
        sentences_demographics = data_df.iloc[:,-1].values

        # method 13
        messages_big5_training = sentences_embeddings
        if "big5" in args.train_data_file:
            dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        elif "depanganx" in args.train_data_file:
            dimensions_name = ["depression",  "anger", "anxiety"]
        elif "life" in args.train_data_file or "swl" in args.train_data_file:
            dimensions_name = ["swl"]
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # # set parameters
        # explore_std_range = [-4.0,4.2]
        # std_step_interval = 0.5
        # std_random_level = 0.00001
        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
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
                print("===============")
                print("====> Samples around mean + {}*std:".format(round(std_position,2)))

                # explore the demographics variables
                demographic_type = data_df.columns[-1]
                if demographic_type=='age':
                    # demographic_std_step = [0.0] # experiment fix demographic, change psychological variable
                    demographic_std_step = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0] # experiment fix psychological variable, change demographic
                    # demographic_std_step = [-2.0, 0.0, 2.0, 3.0] # works for dep, when fix dep and run age alone, 
                    # demographic_std_step = [-3.0, 0, 3.0] # => works for swl
                    demographic_age_mean = sentences_demographics.mean()
                    demographic_age_std = sentences_demographics.std()
                    demographic_values = [demographic_age_mean + demographic_age_std*item for item in demographic_std_step]
                elif demographic_type=='gender':
                    demographic_std_step = [0.0] # experiment fix demographic, change psychological variable
                    # demographic_std_step = [-5.0,-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0] # experiment fix psychological variable, change demographic
                    # demographic_std_step = [-4.0,4.0] # => works for swl
                    # demographic_std_step = [-0.9,0.9] # => works for dep 
                    demographic_std_step = [-1.0,1.0]
                    demographic_gender_mean = sentences_demographics.mean()
                    demographic_gender_std = sentences_demographics.std()
                    demographic_values = [demographic_gender_mean + demographic_gender_std*item for item in demographic_std_step]
                for demographic_value in demographic_values:
                    if demographic_type=='age':
                        print("\n")
                        print("=> Demographic variable [{}] - value [{}]: ".format(demographic_type, str(demographic_value)))   
                    elif demographic_type=='gender':
                        print("\n")
                        print("=> Demographic variable [{}] - value [{} ({})]: ".format(demographic_type, "Male" if demographic_value<0 else "Female", str(demographic_value)))   
                    
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
                            generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_14":
        # generate texts with input is message-level big5 score
        # for each pair of personalities, chooing +1/-1 for one and +1/-1 for the other
        # e.g.: generate text for message-level [0,0,+2.std,0,-2.std], [0,-2.std,0,0,+2.std]

        # method 14
        messages_big5_training = sentences_embeddings
        if "big5" in args.train_data_file:
            dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        elif "depanganx" in args.train_data_file:
            dimensions_name = ["depression",  "anger", "anxiety"]
        elif "life" in args.train_data_file or "swl" in args.train_data_file:
            dimensions_name = ["swl"]        
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)

        # # set parameters
        # explore_std_range = [-3.0,3.2]
        # std_step_interval = args.generate_interval
        # std_random_level = 0.0000
        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
        std_random_level = 0.00001

        
        # generate sentence for each hidden dimension
        hidden_size = means.shape[0]
        for i in range(hidden_size):
            for j in range(i+1, hidden_size):

                if i==j:
                    continue

                print("\n")
                print("=====")
                print("HIDDEN DIMENISON {} ({}) and {} ({}):".format(str(i), dimensions_name[i], str(j), dimensions_name[j]))   

                generated_texts_save = {}
                for case in [[-1,-1],[1,-1],[1,1],[-1,1]]:

                    ##### generating texts
                    print("\n")
                    print("generation avoid repeating!")
                    print("generate for position {}*std for {} ({}) and {}*std for {} ({}): ".format(str(case[0]*std_step_interval), str(i), dimensions_name[i], str(case[1]*std_step_interval), str(j), dimensions_name[j]))
                    generated_samples = []   # avoid repeated generated_sample
                    for _ in range(args.generate_num):   

                        # sample embedding around embedding + std_position*stds[i]
                        epsilon = np.random.uniform(-std_random_level,std_random_level)
                        message_big5 = np.copy(means)
                        message_big5[i] = message_big5[i] + case[0]*std_step_interval*stds[i] + epsilon*stds[i]
                        message_big5[j] = message_big5[j] + case[1]*std_step_interval*stds[j] + epsilon*stds[j]

                        # concat (user_big5 - not applicable) and message_big5
                        embedding_sample = message_big5

                        # transform to tensor
                        embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
            
                        # generate sentence
                        generated_count = 0    
                        while True:
                            generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


                    # save to dictionary then save to csv
                    column_name = "\"{}*std_{}_and_{}*std_{}\"".format(str(case[0]*std_step_interval), dimensions_name[i], str(case[1]*std_step_interval), dimensions_name[j])
                    generated_texts_save[column_name] = generated_samples

                # print to csv
                write_df = pd.DataFrame.from_dict(generated_texts_save)
                write_df.to_csv(args.csv_path + "/{}_{}.csv".format(dimensions_name[i],dimensions_name[j]), header=True, index=False)


    if args.method == "method_15.1":
        # generate texts with input is user-level big5 score, with prompting text
        # for each personalities, travel along the dimension of that personality to generate text, conditioned on big
        # 5 user-level vector and a prompting text
        # e.g.: generate text for user-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]

        # method 15.1
        # get prompting text
        prompting_text = 'the weather is'

        # conditioned on the prompting_text, travel along the personality poles ("ope",  "con", "ext", "agr", "neu")   
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # # set parameters
        # explore_std_range = [-4.0,4.2]
        # std_step_interval = 4
        # std_random_level = 0.00001
        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
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
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(prompting_text = prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_15.2":
        # generate texts with input is message-level big5 score, with prompting text
        # for each personalities, travel along the dimension of that personality to generate text, conditioned on big
        # 5 message-level vector and a prompting text
        # e.g.: generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]

        # method 15.2
        ## get prompting text
        # prompting_text = 'the weather is'
        # prompting_text = 'today is'
        # prompting_text = 'love is'
        # prompting_text = 'tomorrow I will'
        # prompting_text = 'I hate'
        # prompting_text = 'parties'
        # prompting_text = 'social media'
        # prompting_text = 'life is'
        # prompting_text = 'today is'
        # prompting_text = 'I like to'
        prompting_text = args.prompting_text

        # conditioned on the prompting_text, travel along the personality poles ("ope",  "con", "ext", "agr", "neu")   
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
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

                # no need to generate at 0 position
                if std_position==0:
                    continue
            
                print("\n")
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(prompting_text = prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_16":
        # this method return the likelihood of a sentence, given the input 5 vector
        # e.g.: with the input vector "0,0,2,0,0", what is the likelihood of "I want to ride a bike"

        # utilities methods
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
        def likelihood(decoder_lm_logits, target, ignore_index):
            # Negative Log Likelihood
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
            # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
            NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  
            # transform from NLL to likelihood
            likelihood = torch.exp((-1)*NLL_loss)
            return likelihood
        def truncating_padding_sentence(tokens, block_size):
            if (len(tokens) > block_size):
                original_tokens_len = block_size
                tokens = tokens[:block_size]
            else:
                original_tokens_len = len(tokens)
                tokens = tokens + ["<|pad|>"]*(block_size - len(tokens))
            return tokens, original_tokens_len    
        def probability(decoder_lm_logits, target, ignore_index):
            # Negative Log Likelihood
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
            # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
            NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  
            # transform from NLL to likelihood
            print("target.contiguous().view(-1).shape: " + str(target.contiguous().view(-1).shape[0]))
            print("NLL_loss: " + str(NLL_loss))
            probability = torch.exp((-1)*NLL_loss)
            return probability

        # set dimensions
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        messages_big5_training = sentences_embeddings
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # # set parameters
        # explore_std_range = [-6.0,6.2]
        # std_step_interval = 0.5
        # std_random_level = 0
        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
        std_random_level = 0.00001


        # loop through each dimension
        for i in range(len(dimensions_name)):
            print("\n")
            print("=====")
            print("HIDDEN DIMENISON {} ({}) : ".format(str(i), dimensions_name[i]))   

            ##### calculating likelihood
            # loop along the pole
            sentence_embeddings = []
            sentence_embeddings_std = []
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):

                # sample embedding around embedding + std_position*stds[i]
                epsilon = np.random.uniform(-std_random_level,std_random_level)
                sentence_embedding_oneposition = np.copy(means)
                sentence_embedding_oneposition[i] =  sentence_embedding_oneposition[i] + std_position*stds[i] + epsilon*stds[i]
                sentence_embeddings.append(sentence_embedding_oneposition)
                sentence_embeddings_std.append(std_position)

            sentence_texts = args.evaluating_text
            sentence_texts = sentence_texts.split("|")
            # print(sentence_embeddings)
            # print(sentence_texts)

            # going through each pair of sentence_embedding and sentence_text
            likelihood_list = []
            for sentence_embedding in sentence_embeddings:
                for sentence_text in sentence_texts:

                    # tokenize sentence
                    sentence_tokenized = model.tokenizer.tokenize(sentence_text)

                    # decoder_input
                    decoder_input = ["<|sos|>"] + sentence_tokenized
                    decoder_input, decoder_input_len = truncating_padding_sentence(decoder_input, args.block_size)
                    decoder_input = model.tokenizer.convert_tokens_to_ids(decoder_input)
                    decoder_input = np.array(decoder_input)
                    # decoder_output
                    decoder_label = sentence_tokenized + ["<|endoftext|>"]
                    decoder_label, decoder_label_len = truncating_padding_sentence(decoder_label, args.block_size)
                    decoder_label = model.tokenizer.convert_tokens_to_ids(decoder_label)
                    decoder_label = np.array(decoder_label)
                    # put into batch_size of 1
                    sentence_embedding = np.array([sentence_embedding])
                    decoder_input = np.array([decoder_input])
                    decoder_label = np.array([decoder_label])
            
                    # create decoder_attention_mask here
                    decoder_attention_mask_onesample = create_attention_mask(0, args.block_size, args.gpt2_config, "decoder_mask")

                    # transform tensor to correct type
                    sentence_embedding = torch.from_numpy(sentence_embedding).float()
                    decoder_input = torch.from_numpy(decoder_input).long() 
                    decoder_label = torch.from_numpy(decoder_label).long()        
                    decoder_attention_mask = torch.tensor([decoder_attention_mask_onesample] * 1).long() 
                    sentence_embedding = sentence_embedding.to(args.device)
                    decoder_input = decoder_input.to(args.device)
                    decoder_label = decoder_label.to(args.device)
                    decoder_attention_mask = decoder_attention_mask.to(args.device)

                    # forward pass (change and edit with VAE code)
                    # decoder_lm_logits = model(sentence_embedding, decoder_input, decoder_attention_mask, args.device)
                    decoder_lm_logits = model(sentence_embedding, decoder_input, None, args.device)

                    # compute likelihood or probability
                    likelihood_value = likelihood(decoder_lm_logits, decoder_label, model.tokenizer.convert_tokens_to_ids(["<|pad|>"])[0])    
                    ## likelihood_value = probability(decoder_lm_logits, decoder_label, model.tokenizer.convert_tokens_to_ids(["<|pad|>"])[0])    
                    likelihood_value = likelihood_value.data.cpu().numpy()
                    likelihood_list.append(np.round(likelihood_value,6))

                    # # show results
                    # print("Input embeddings: " + str(sentence_embedding))
                    # print("Evaluating text: " + str(sentence_text))
                    # print("Likelihood: " + str(np.round(likelihood_value,6)))
                    # print("========")
            # print("Text: " + sentence_text)
            # print("List of sentence_embeddings:")
            # print([item[i] for item in sentence_embeddings])
            # print("List of sentence_embeddings_std:")
            # print(sentence_embeddings_std)
            # print("List of likelihoods: ")
            # print(likelihood_list)

            print("text = \"{}\"".format(sentence_text))
            print("dimension = \"{}\"".format(dimensions_name[i]))
            print("x = {}".format(str(sentence_embeddings_std)))
            print("y = {}".format(likelihood_list))

            # plot 
            print(dimensions_name[i])
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure()
            ax = plt.axes()
            text = sentence_text
            dimension = dimensions_name[i]
            x = sentence_embeddings_std
            y = likelihood_list
            plt.figure(figsize=(8, 6)).add_subplot(1, 1, 1).plot(x, y)
            plt.axvline(x=0.0,color='r', ls='--')
            plt.ylabel("likelihood", fontsize=20)
            plt.xlabel("{} (+/-std)".format(dimension), fontsize=20)
            plt.xlim((-4,4))
            plt.title("Text: \"{}\"".format(text),fontweight='bold', fontsize=20)


    if args.method == "method_17":
        # this method compute the likelihood of items in the big5 questionnaire, then classify 
        # those items into the right key and construct


        # utilities methods
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
        def likelihood(decoder_lm_logits, target, ignore_index):
            # Negative Log Likelihood
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
            # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
            NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  
            # transform from NLL to likelihood
            likelihood = torch.exp((-1)*NLL_loss)
            return likelihood
        def truncating_padding_sentence(tokens, block_size):
            if (len(tokens) > block_size):
                original_tokens_len = block_size
                tokens = tokens[:block_size]
            else:
                original_tokens_len = len(tokens)
                tokens = tokens + ["<|pad|>"]*(block_size - len(tokens))
            return tokens, original_tokens_len    
        def probability(decoder_lm_logits, target, ignore_index):
            # Negative Log Likelihood
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index) # this 'mean' is taking average across all predicted tokens: sum(crossentropyloss_each_position)/(batch_size * seq_length)
            # transform decoder_lm_logits from [batch_size, seq_length, vocab_size] => [batch_size * seq_length, vocab_size], target from [batch_size, sweq_length] => [batch_size * sweq_length]
            NLL_loss = loss_fct(decoder_lm_logits.view(-1, decoder_lm_logits.size(-1)), target.contiguous().view(-1))  
            # transform from NLL to likelihood
            print("target.contiguous().view(-1).shape: " + str(target.contiguous().view(-1).shape[0]))
            print("NLL_loss: " + str(NLL_loss))
            probability = torch.exp((-1)*NLL_loss)
            return probability
        def calculate_area_likelihood(likelihood_vector):
           mid_point = int(np.floor(len(likelihood_vector)/2))
           area_likelihood_vector = [np.sum(likelihood_vector[:mid_point]), likelihood_vector[mid_point], np.sum(likelihood_vector[(mid_point+1):])]
           return area_likelihood_vector


        # set dimensions
        dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        messages_big5_training = sentences_embeddings
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)

        # set sentence_texts and sentence_construct_key
        sentence_texts = ['i have a vivid imagination.', 'i hold a grudge.', 'i do not mind being the centre of attention.', 'i do not like poetry.', 'i complete tasks successfully.', 'i believe that others have good intentions.', 'i avoid philosophical discussions.', 'i need a push to get started.', 'i cut others to pieces.', 'i make friends easily.', 'i feel comfortable with myself.', 'i often feel blue.', 'i am easy to satisfy.', 'i keep in the background.', 'i am always prepared.', 'i enjoy wild flights of fantasy.', 'i get stressed out easily.', 'i avoid contact with others.', 'i am not easily bothered by things.', 'i shirk my duties.', 'i can say things beautifully.', 'i suspect hidden motives in others.', 'i cheer people up.', 'i am not interested in abstract ideas.', 'i do things according to a plan.', 'i am concerned about others.', 'i am relaxed most of the time.', 'i waste my time.', 'i dont talk a lot.', 'i dislike myself.', 'i enjoy thinking about things.', 'i get back at others.', 'i warm up quickly to others.', 'i do not enjoy going to art museums.', 'i follow through with my plans.', 'i make people feel at ease.', 'i seldom feel blue.', 'i mess things up.', 'i have little to say.', 'i fear for the worst.', 'i carry the conversation to a higher level.', 'i believe that i am better than others.', 'i talk to a lot of different people at parties.', 'i tend to vote for conservative political candidates.', 'i am exacting in my work.', 'i accept people as they are.', 'i am not easily frustrated.', 'i dont put my mind on the task at hand.', 'i keep others at a distance.', 'i panic easily.', 'i tend to vote for liberal political candidates.', 'i contradict others.', 'i know how to captivate people.', 'i am not interested in theoretical discussions.', 'i make plans and stick to them.', 'i trust what people say.', 'i rarely lose my composure.', 'i leave things unfinished.', 'i dont like to draw attention to myself.', 'i worry about things.', 'i enjoy hearing new ideas.', 'i make demands on others.', 'i am the life of the party.', 'i have difficulty understanding abstract ideas.', 'i finish what i start.', 'i respect others.', 'i rarely get irritated.', 'i make a mess of things.', 'i find it difficult to approach others.', 'i have frequent mood swings.', 'i have a rich vocabulary.', 'i am out for my own personal gain.', 'i am skilled in handling social situations.', 'i do not like art.', 'i get chores done right away.', 'i sympathise with others feelings.', 'i seldom get mad.', 'i dont see things through.', 'i retreat from others.', 'i am often down in the dumps.', 'i get excited by new ideas.', 'i have a sharp tongue.', 'i feel comfortable around people.', 'i believe that too much tax money goes to support artists.', 'i pay attention to details.', 'i have a good word for everyone.', 'i remain calm under pressure.', 'i find it difficult to get down to work.', 'i would describe my experiences as somewhat dull.', 'i am filled with doubts about things.', 'i believe in the importance of art.', 'i insult people.', 'i start conversations.', 'i rarely look for a deeper meaning in things.', 'i carry out my plans.', 'i treat all people equally.', 'i am very pleased with myself.', 'i do just enough work to get by.', 'i am hard to get to know.', 'i feel threatened easily.']
        sentence_construct_key = ['1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '-1:O', '-1:C', '-1:A', '1:E', '1:N', '-1:N', '1:A', '-1:E', '1:C', '1:O', '-1:N', '-1:E', '1:N', '-1:C', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N', '1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '1:N', '-1:C', '-1:E', '-1:N']
        # sentence_texts = ['i have a vivid imagination.', 'i hold a grudge.', 'i do not mind being the centre of attention.', 'i do not like poetry.', 'i complete tasks successfully.', 'i believe that others have good intentions.', 'i avoid philosophical discussions.', 'i need a push to get started.', 'i cut others to pieces.', 'i make friends easily.']
        # sentence_construct_key = ['1:O', '-1:A', '1:E', '-1:O', '1:C', '1:A', '-1:O', '-1:C', '-1:A', '1:E']
        # sentence_texts = ['i like art', 'i do not like art', 'I dislike art', 'I hate art', 'I hate people', 'I do not hate people', 'I love people']
        # sentence_construct_key = ['1:O', '-1:O', '-1:O', '-1:O', '-1:A', '1:A', '1:A']

        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
        std_random_level = 0.00001


        # loop through all items
        items_results = []
        for sentence_id, sentence_text in enumerate(sentence_texts):
            print("Item number {}:".format(str(sentence_id + 1)))

            # loop through each dimension
            one_item_results = dict()
            for i in range(len(dimensions_name)):
                # loop along the pole create embeddings
                sentence_embeddings = []
                sentence_embeddings_std = []
                for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    sentence_embedding_oneposition = np.copy(means)
                    sentence_embedding_oneposition[i] =  sentence_embedding_oneposition[i] + std_position*stds[i] + epsilon*stds[i]
                    sentence_embeddings.append(sentence_embedding_oneposition)
                    sentence_embeddings_std.append(std_position)
                
                # loop along the pole compute likelihood
                likelihood_list = []
                for sentence_embedding in sentence_embeddings:
                # tokenize sentence
                    sentence_tokenized = model.tokenizer.tokenize(sentence_text)

                    # decoder_input
                    decoder_input = ["<|sos|>"] + sentence_tokenized
                    decoder_input, decoder_input_len = truncating_padding_sentence(decoder_input, args.block_size)
                    decoder_input = model.tokenizer.convert_tokens_to_ids(decoder_input)
                    decoder_input = np.array(decoder_input)
                    # decoder_output
                    decoder_label = sentence_tokenized + ["<|endoftext|>"]
                    decoder_label, decoder_label_len = truncating_padding_sentence(decoder_label, args.block_size)
                    decoder_label = model.tokenizer.convert_tokens_to_ids(decoder_label)
                    decoder_label = np.array(decoder_label)
                    # put into batch_size of 1
                    sentence_embedding = np.array([sentence_embedding])
                    decoder_input = np.array([decoder_input])
                    decoder_label = np.array([decoder_label])
            
                    # create decoder_attention_mask here
                    decoder_attention_mask_onesample = create_attention_mask(0, args.block_size, args.gpt2_config, "decoder_mask")

                    # transform tensor to correct type
                    sentence_embedding = torch.from_numpy(sentence_embedding).float()
                    decoder_input = torch.from_numpy(decoder_input).long() 
                    decoder_label = torch.from_numpy(decoder_label).long()        
                    decoder_attention_mask = torch.tensor([decoder_attention_mask_onesample] * 1).long() 
                    sentence_embedding = sentence_embedding.to(args.device)
                    decoder_input = decoder_input.to(args.device)
                    decoder_label = decoder_label.to(args.device)
                    decoder_attention_mask = decoder_attention_mask.to(args.device)

                    # forward pass (change and edit with VAE code)
                    # decoder_lm_logits = model(sentence_embedding, decoder_input, decoder_attention_mask, args.device)
                    decoder_lm_logits = model(sentence_embedding, decoder_input, None, args.device)

                    # compute likelihood or probability
                    likelihood_value = likelihood(decoder_lm_logits, decoder_label, model.tokenizer.convert_tokens_to_ids(["<|pad|>"])[0])    
                    ## likelihood_value = probability(decoder_lm_logits, decoder_label, model.tokenizer.convert_tokens_to_ids(["<|pad|>"])[0])    
                    likelihood_value = likelihood_value.data.cpu().numpy()
                    likelihood_list.append(np.round(likelihood_value,6))

                # # print out results for this dimension
                # print("text = \"{}\" ({})".format(sentence_text, sentence_construct_key[sentence_id]))
                # print("dimension = \"{}\"".format(dimensions_name[i]))
                # print("x = {}".format(str(sentence_embeddings_std)))
                # print("y = {}".format(likelihood_list))

                one_item_results[dimensions_name[i]] = likelihood_list
            
            items_results.append(one_item_results)
            print("================")


        # plot figure or not 
        figure_plot = False
        if figure_plot:
            # plot figures for each item
            for sentence_id, sentence_text in enumerate(sentence_texts):
                one_item_results = items_results[sentence_id]
                one_item_construct_key = sentence_construct_key[sentence_id]

                item_construct = one_item_construct_key.split(":")[1]
                construct_lookup = {'O': "openness",  'C': "conscientiousness", 'E':"extroversion", 'A':"agreeableness", 'N':"neuroticism"}
                item_key = one_item_construct_key.split(":")[0]

                likelihood_vector = one_item_results[construct_lookup[item_construct]]

                x = np.arange(explore_std_range[0], explore_std_range[1], std_step_interval)
                y = likelihood_vector
                dimension = construct_lookup[item_construct]
                text = sentence_text

                # plot 
                import matplotlib.pyplot as plt
                plt.style.use('seaborn-whitegrid')
                ax = plt.axes()
                plt.figure(figsize=(8, 6)).add_subplot(1, 1, 1).plot(x, y);
                plt.axvline(x=0.0,color='r', ls='--')
                plt.ylabel("likelihood", fontsize=20)
                plt.xlabel("{} (+/-std)".format(dimension), fontsize=20)
                plt.xlim((-4,4))
                plt.title("Text: \"{}\"".format(text),fontweight='bold', fontsize=20)
                figname = "item_"+str(sentence_id+1)
                plt.savefig(args.figures_dir + figname)

        
        # parsing all items results
        parse_results = True
        if parse_results:
            test_1_results = []
            test_2_results = []
            test_2_details_results = {'groundtruth':[],'prediction':[]}
            test_3_results = []
            test_3_details_results = {'groundtruth':[],'prediction':[]}
            construct_list = []
            for sentence_id, sentence_text in enumerate(sentence_texts):
                one_item_results = items_results[sentence_id]
                one_item_construct_key = sentence_construct_key[sentence_id]

                item_construct = one_item_construct_key.split(":")[1]
                construct_list.append(item_construct)
                construct_lookup = {'O': "openness",  'C': "conscientiousness", 'E':"extroversion", 'A':"agreeableness", 'N':"neuroticism"}
                item_key = one_item_construct_key.split(":")[0]

                # test 1.1 (high and low of the dimension):
                if True:
                    likelihood_vector = one_item_results[construct_lookup[item_construct]]  # likelihood_vector = np.random.random(3) # for randomize
                    area_likelihood_vector = calculate_area_likelihood(likelihood_vector) 
                    if item_key=='1':
                        if area_likelihood_vector[-1] > area_likelihood_vector[0]:
                            test_1_results.append(True)
                        else:
                            test_1_results.append(False)
                    elif item_key=='-1':
                        if area_likelihood_vector[-1] < area_likelihood_vector[0]:
                            test_1_results.append(True)
                        else:
                            test_1_results.append(False)
                    else:
                        exit()
                    # # DEBUGGING!    
                    # print(sentence_text)
                    # print(one_item_construct_key)
                    # print(likelihood_vector)
                    # print(test_1_results[-1])
                    # print("---")


                # test 2 (predict the exact key and construct among high, low of dimensions) results):
                if True:
                    max_likelihood = 0
                    max_construct = None
                    max_direction = None
                    for construct in construct_lookup.keys():
                        # compute area likelihood
                        likelihood_vector = one_item_results[construct_lookup[construct]] # likelihood_vector = np.random.random(3) # for randomize
                        area_likelihood_vector = calculate_area_likelihood(likelihood_vector) 
                        # looking through the direction
                        for direction in [0,-1]:
                            if area_likelihood_vector[direction] > max_likelihood:
                                max_likelihood = area_likelihood_vector[direction]
                                max_construct = construct
                                max_direction = direction    
                    if max_direction==0:
                        max_direction = '-1'
                    elif max_direction==-1:
                        max_direction = '1'
                    if (max_construct==item_construct) and (max_direction==item_key):
                        test_2_results.append(True)
                    else:
                        test_2_results.append(False)
                    # recording results details
                    test_2_details_results['groundtruth'].append(item_construct+':'+item_key)
                    test_2_details_results['prediction'].append(max_construct+':'+max_direction)
                    # DEBUGGING!    
                    print(sentence_text)
                    # print(one_item_construct_key)
                    # print(one_item_results)
                    # print("max_construct: {}, max_direction: {}".format(max_construct, max_direction))
                    # print("item_construct: {}, item_key: {}".format(item_construct, item_key))
                    # print(test_2_results[-1])
                    # print("---")
                    print("groundtruth ({}:{}) - prediction ({}:{})".format(item_construct, item_key, max_construct, max_direction))
                    print("---")

                # test 3 (predict the exact construct) results):
                if True:
                    max_likelihood = 0
                    max_construct = None
                    for construct in construct_lookup.keys():
                        # compute area likelihood
                        likelihood_vector = one_item_results[construct_lookup[construct]] # likelihood_vector = np.random.random(3) # for randomize
                        area_likelihood = np.sum(likelihood_vector)
                        if area_likelihood > max_likelihood:
                            max_likelihood = area_likelihood
                            max_construct = construct
                    if (max_construct==item_construct):
                        test_3_results.append(True)
                    else:
                        test_3_results.append(False)
                    # recording results details
                    test_3_details_results['groundtruth'].append(item_construct)
                    test_3_details_results['prediction'].append(max_construct)


            # report results
            print("Summary results:")
            print("test 1 (high and low of the dimension) results: " + str(np.sum(test_1_results)/len(test_1_results)))
            print("test 2 (predict the exact key and construct) results: " + str(np.sum(test_2_results)/len(test_2_results)))
            print("test 2 (predict the exact key and construct) results: " + str(np.sum(test_3_results)/len(test_3_results)))
            print("\n")
            # fine grained results:
            print("Details results:")
            print("test 1: ")
            for construct in set(construct_list):
                construct_count = np.sum([item==construct for item in construct_list])
                construct_indices = [i for i, x in enumerate(construct_list) if x == construct]
                construct_correct = np.sum([test_1_results[index] for index in construct_indices])
                print("{} - {}/{}".format(construct, str(construct_correct), str(construct_count)))
            print("test 2: ")
            from sklearn.metrics import confusion_matrix
            labels = ['O:1','O:-1','C:1','C:-1','E:1','E:-1','A:1','A:-1','N:1','N:-1']
            confusion_matrix = confusion_matrix(test_2_details_results['groundtruth'], test_2_details_results['prediction'], labels = labels)
            print(confusion_matrix)
            for i in range(len(labels)):
                print("groundtruth label: {}".format(labels[i]))
                print("predicted label: " + str([(labels[j] + " => "+ str(confusion_matrix[i][j])) for j in range(len(labels))]))
                print("---")
            print("test 3: ")
            from sklearn.metrics import confusion_matrix
            labels = ['O','C','E','A','N']
            confusion_matrix = confusion_matrix(test_3_details_results['groundtruth'], test_3_details_results['prediction'], labels = labels)
            print(confusion_matrix)
            for i in range(len(labels)):
                print("groundtruth label: {}".format(labels[i]))
                print("predicted label: " + str([(labels[j] + " => "+ str(confusion_matrix[i][j])) for j in range(len(labels))]))
                print("---")


    if args.method == "method_18":
        # generate texts with input is message-level big5 score, then write to csv for human evaluation
        # for each personalities, travel along the dimension of that personality to generate text
        # e.g.: generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]
        
        # method 18
        messages_big5_training = sentences_embeddings
        if "big5" in args.train_data_file:
            dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        elif "depanganx" in args.train_data_file:
            dimensions_name = ["depression",  "anger", "anxiety"]
        elif "life" in args.train_data_file or "swl" in args.train_data_file:
            dimensions_name = ["swl"]        
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # # set parameters
        # explore_std_range = [-4.0,4.2]
        # std_step_interval = 0.5
        # std_random_level = 0.00001
        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
        std_random_level = 0.00001


        # generate sentence for each hidden dimension
        all_generated_texts = {}
        hidden_size = means.shape[0]
        for i in range(hidden_size):
        
            # save all generated texts for each dimension
            all_generated_texts[dimensions_name[i]] = []

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
                        generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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

                    # save generated texts
                    all_generated_texts[dimensions_name[i]].append([generated_sample_clean, std_position])


        # save to csv for each dimension
        for dimension in dimensions_name:
            # rearrange sentences
            df = pd.DataFrame(all_generated_texts[dimension], columns = ['generated_sentence','std_position'])
            std_positions = list(set(df['std_position']))
            organized_texts = []
            for std_position in std_positions:
                df_std_position = df.loc[df['std_position']==std_position]
                def flow_from_df(dataframe: pd.DataFrame, chunk_size: int = 5):
                    chunks = []
                    for start_row in range(0, dataframe.shape[0], chunk_size):
                        end_row  = min(start_row + chunk_size, dataframe.shape[0])
                        chunks.append(dataframe.iloc[start_row:end_row, :].values)
                    return chunks
                chunks = flow_from_df(df_std_position, 5)
                organized_texts.extend(chunks)


            # shuffling list organized_texts
            import random
            random.shuffle(organized_texts)
            random.seed(args.seed)
            # print("organized_texts:")
            # print(organized_texts)


            # open csv_file and write
            write_list = []
            for chunk in organized_texts:
                for item in chunk:
                    item[0] = item[0].replace("<|sos|>","").replace("<|endoftext|>","")
                    write_list.append(item)
                    print("item: " + str(item))
                write_list.append(["",""])
            write_df = pd.DataFrame(write_list, columns=['generated_text', 'std_position'])
            print(write_df)
            write_df.to_csv(args.csv_path + "/{}_generated.csv".format(dimension), header=True, index=False)


    if args.method == "method_19":
        # generate texts with input is message-level big5 score, with prompting text
        # for each personalities, travel along the dimension of that personality to generate text, conditioned on big
        # 5 message-level vector and a prompting text
        # e.g.: generate text for message-level [0,0,-2.std,0,0], [0,0,-1.std,0,0], [0,0,0,0,0], [0,0,+1.std,0,0], [0,0,+2.std,0,0]

        # method 15.2
        ## get prompting text
        # prompting_text = 'the weather is'
        # prompting_text = 'today is'
        # prompting_text = 'love is'
        # prompting_text = 'tomorrow I will'
        # prompting_text = 'I hate'
        # prompting_text = 'parties'
        # prompting_text = 'social media'
        # prompting_text = 'life is'
        # prompting_text = 'today is'
        # prompting_text = 'I like to'
        prompting_text = args.prompting_text

        # conditioned on the prompting_text, travel along the personality poles ("ope",  "con", "ext", "agr", "neu")   
        if "big5" in args.train_data_file:
            dimensions_name = ["openness",  "conscientiousness", "extroversion", "agreeableness", "neuroticism"]
        elif "depanganx" in args.train_data_file:
            dimensions_name = ["depression",  "anger", "anxiety"]
        elif "life" in args.train_data_file or "swl" in args.train_data_file:
            dimensions_name = ["swl"]        
        means = np.mean(sentences_embeddings, axis = 0)
        stds = np.std(sentences_embeddings, axis = 0)


        # set parameters
        explore_std_range = [-args.std_range,args.std_range+0.01]
        std_step_interval = args.generate_interval
        std_random_level = 0.00001


        # generate sentence for each hidden dimension
        all_generated_texts = {}
        hidden_size = means.shape[0]
        for i in range(hidden_size):

            # save all generated texts for each dimension
            all_generated_texts[dimensions_name[i]] = []

            print("\n")
            print("=====")
            print("HIDDEN DIMENISON {} ({}) : ".format(str(i), dimensions_name[i]))   
                   
            ##### generating texts
            print("***** generating text in interval: ")
            # explore the pole
            for std_position in np.arange(explore_std_range[0], explore_std_range[1], std_step_interval):

                # no need to generate at 0 position
                if std_position==0:
                    continue
            
                print("\n")
                print("samples around mean + {}*std:".format(round(std_position,1)))
                print("generation avoid repeating!")
                generated_samples = []   # avoid repeated generated_sample
                for _ in range(args.generate_num):     
                    
                    # sample embedding around embedding + std_position*stds[i]
                    epsilon = np.random.uniform(-std_random_level,std_random_level)
                    embedding_sample = np.copy(means)
                    embedding_sample[i] = embedding_sample[i] + std_position*stds[i] + epsilon*stds[i]
                    embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
           
                    # generate sentence
                    generated_count = 0    
                    while True:
                        generated_sample, decoder_attentions_sample = model.inference(prompting_text = prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
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

                    # save generated texts
                    all_generated_texts[dimensions_name[i]].append([generated_sample_clean, std_position])


        # save to csv for each dimension
        for dimension in dimensions_name:
            # rearrange sentences
            df = pd.DataFrame(all_generated_texts[dimension], columns = ['generated_sentence','std_position'])
            std_positions = list(set(df['std_position']))
            organized_texts = []
            for std_position in std_positions:
                df_std_position = df.loc[df['std_position']==std_position]
                def flow_from_df(dataframe: pd.DataFrame, chunk_size: int = 5):
                    chunks = []
                    for start_row in range(0, dataframe.shape[0], chunk_size):
                        end_row  = min(start_row + chunk_size, dataframe.shape[0])
                        chunks.append(dataframe.iloc[start_row:end_row, :].values)
                    return chunks
                chunks = flow_from_df(df_std_position, 5)
                organized_texts.extend(chunks)


            # shuffling list organized_texts
            import random
            random.shuffle(organized_texts)
            random.seed(args.seed)
            # print("organized_texts:")
            # print(organized_texts)


            # open csv_file and write
            write_list = []
            for chunk in organized_texts:
                for item in chunk:
                    item[0] = item[0].replace("<|sos|>","").replace("<|endoftext|>","")
                    write_list.append(item)
                    print("item: " + str(item))
                write_list.append(["",""])
            write_df = pd.DataFrame(write_list, columns=['generated_text', 'std_position'])
            print(write_df)
            write_df.to_csv(args.csv_path + "/{}_generated_prompts.csv".format(dimension), header=True, index=False)


    if args.method == "method_20":
        # generate texts around the interpersonal-circumplex circle
        # the interpersonal-circumplex is 

        # method 20
        messages_big5_training = sentences_embeddings
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)


        # list of positions on the circumplex to generate texts from, then transform it to big5 positions
        position_list = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]  # 1st index is Warmth, 2nd index is Dominance
        diagnol = 1/np.sqrt(2)
        position_list = [[0,1],[diagnol,diagnol],[1,0],[diagnol,-diagnol],[0,-1],[-diagnol,-diagnol],[-1,0],[-diagnol,diagnol]]  # 1st index is Warmth, 2nd index is Dominance
        position_name = ["Assured-Dominant", "Gregarious-Extraverted", "Warm-Agreeable", "Unassuming-Ingenuous", "Unassured-Submissive", "Aloof-Introverted", "Cold-Hearted", "Arrogant-Calculating"]
        def rotate(origin, point, angle):
            import math
            ox, oy = origin
            px, py = point
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy
        transformed_position_list = [rotate([0,0],position,math.radians(22.5)) for position in position_list] # 1st index is Agr, 2nd index is Ext
        print("position_list:")
        print(position_list)
        print("transformed_position_list:")
        print(transformed_position_list)



        # loop through all position and generate texts
        for i in range(len(position_list)):
            position = position_list[i]
            area_name = position_name[i]
            transformed_position = transformed_position_list[i]
            print("\n=======")
            print("generation avoid repeating!")
            print("Warmth-Dominance {} = Agr-Ext {} ({})".format(str(position),str(transformed_position), str(area_name)))
            generated_samples = []   # avoid repeated generated_sample
            for _ in range(args.generate_num):    

                # sample embedding around embedding + std_position*stds[i]
                std_random_level = 0.00001
                std_position1 = 4.0
                std_position2 = 3.0
                epsilon = np.random.uniform(-std_random_level,std_random_level)
                message_big5 = np.copy(means)
                message_big5[3] = message_big5[3] + transformed_position[0]*stds[3]*std_position1 + epsilon*stds[3]
                message_big5[2] = message_big5[2] + transformed_position[1]*stds[2]*std_position2 + epsilon*stds[2]

                # concat (user_big5 - not applicable) and message_big5
                embedding_sample = message_big5

                # transform to tensor
                embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
        
                # generate sentence
                generated_count = 0    
                while True:
                    generated_sample, decoder_attentions_sample = model.inference(sentence_embedding = embedding_sample, args = args, device = device)
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


    if args.method == "method_21":
        # generate texts around the interpersonal-circumplex circle
        # the interpersonal-circumplex is 

        # method 20
        messages_big5_training = sentences_embeddings
        means = np.mean(messages_big5_training, axis = 0)
        stds = np.std(messages_big5_training, axis = 0)
        prompting_text = args.prompting_text


        # list of positions on the circumplex to generate texts from, then transform it to big5 positions
        position_list = [[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]]  # 1st index is Warmth, 2nd index is Dominance
        diagnol = 1/np.sqrt(2)
        position_list = [[0,1],[diagnol,diagnol],[1,0],[diagnol,-diagnol],[0,-1],[-diagnol,-diagnol],[-1,0],[-diagnol,diagnol]]  # 1st index is Warmth, 2nd index is Dominance
        position_name = ["Assured-Dominant", "Gregarious-Extraverted", "Warm-Agreeable", "Unassuming-Ingenuous", "Unassured-Submissive", "Aloof-Introverted", "Cold-Hearted", "Arrogant-Calculating"]
        def rotate(origin, point, angle):
            import math
            ox, oy = origin
            px, py = point
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy
        transformed_position_list = [rotate([0,0],position,math.radians(22.5)) for position in position_list] # 1st index is Agr, 2nd index is Ext
        print("position_list:")
        print(position_list)
        print("transformed_position_list:")
        print(transformed_position_list)



        # loop through all position and generate texts
        for i in range(len(position_list)):
            position = position_list[i]
            area_name = position_name[i] 
            transformed_position = transformed_position_list[i]
            print("\n=======")
            print("generation avoid repeating!")
            print("Warmth-Dominance {} = Agr-Ext {} ({})".format(str(position),str(transformed_position), str(area_name)))
            generated_samples = []   # avoid repeated generated_sample
            for _ in range(args.generate_num):    

                # sample embedding around embedding + std_position*stds[i]
                std_random_level = 0.00001
                std_position1 = 4.0
                std_position2 = 2.0
                epsilon = np.random.uniform(-std_random_level,std_random_level)
                message_big5 = np.copy(means)
                message_big5[3] = message_big5[3] + transformed_position[0]*stds[3]*std_position1 + epsilon*stds[3]
                message_big5[2] = message_big5[2] + transformed_position[1]*stds[2]*std_position2 + epsilon*stds[2]

                # concat (user_big5 - not applicable) and message_big5
                embedding_sample = message_big5

                # transform to tensor
                embedding_sample = torch.tensor(embedding_sample, device = device).float().unsqueeze(0)
        
                # generate sentence
                generated_count = 0    
                while True:
                    generated_sample, decoder_attentions_sample = model.inference(prompting_text = prompting_text, sentence_embedding = embedding_sample, args = args, device = device)
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
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.") 
    
    # training parameters
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
        
    # generating parameters
    parser.add_argument("--inference_test", default=0, type=int)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--generate_num", type=int, default=None)
    parser.add_argument("--generate_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--prompting_text", type=str, default="I like to")
    parser.add_argument("--std_range", type=float, default=3.0)
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
    
    if args.inference_test == 1:        
        # Inference test 1 - generate text from each dimension
        inference_test_1(model, args, device)    
    
if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time()

    # total time
    print("Total running time: {} seconds ({} hours)".format(str(round(end_time-start_time)), str(round((end_time-start_time)/3600,2))))      




