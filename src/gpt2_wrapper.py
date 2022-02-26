# import libraries
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import logging
import torch.nn.functional as F
import os
import numpy as np
import pickle

logger = logging.getLogger(__name__)

# ===================== classes of model ===================== #

class GPT2_WRAPPER(nn.Module):

    def __init__(self, gpt2_config, latent_size):
        super(GPT2_WRAPPER, self).__init__()

        # set up transformation matrix and decoder
        self.gpt2_config = gpt2_config
        self.transform_matrix = nn.Linear(latent_size, gpt2_config.n_layer * gpt2_config.n_embd * 2) # CHECK!
        self.tokenizer = None
        self.decoder = None

        # set up gpt2_config
        self.gpt2_config.output_hidden_states = True
        self.gpt2_config.output_past = True
        self.gpt2_config.output_attentions = True


    def initialize_model(self, args):

        # load pretrained model and tokenizer for GPT2 encoder and decoder
        decoder_path = args.gpt2_model_name_or_path   
        tokenizer_path = args.gpt2_model_name_or_path
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)


        # add <|sos|> and <|pad|> to tokenizer
        self.tokenizer.add_special_tokens({"additional_special_tokens":["<|pad|>", "<|sos|>"]})
        self.decoder.resize_token_embeddings(len(self.tokenizer))
        logger.info("tokenizer size: " + str(self.tokenizer.__len__()))
        logger.info("tokenizer.decode [50256, 50257, 50258]: " + str(self.tokenizer.decode([50256, 50257, 50258])) )        


    def forward(self, embeddings, decoder_input_ids, decoder_attention_mask, device):

        # batch_size
        batch_size = embeddings.shape[0]


        # transform to GPT2 transformer size
        transformed_embeddings = self.transform_matrix(embeddings) # CHECK! size [batch_size, n_layer * n_emb * 2]
        # print("transformed_embeddings: " + str(transformed_embeddings.shape)) 
        transformed_embeddings = transformed_embeddings.reshape([batch_size, self.gpt2_config.n_layer, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        # print("transformed_embeddings: " + str(transformed_embeddings.shape)) 
        transformed_embeddings = torch.transpose(transformed_embeddings,0,1).contiguous()
        transformed_embeddings = torch.transpose(transformed_embeddings,1,2).contiguous()
        # print("transformed_embeddings: " + str(transformed_embeddings.shape)) 

        # # switch to match modeling_gpt2.py
        # decoder_attention_mask = torch.transpose(decoder_attention_mask, -1, -2).contiguous()

        # decoder
        past = transformed_embeddings

        # # DEBUGGING!
        # print("past: " + str(past.shape))
        # print("decoder_input_ids: " + str(decoder_input_ids.shape))
        # print("decoder_attention_mask: " + str(decoder_attention_mask.shape))

        # decoder forward pass
        decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = decoder_input_ids, past = past, attention_mask = decoder_attention_mask)

        return decoder_lm_logits


    def inference(self, prompting_text = None, sentence_embedding = None, args = None, device = None):


        # make sure batch_size = 1
        batch_size = sentence_embedding.shape[0]
        assert batch_size == 1


        # transform to GPT2 transformer size
        transformed_embeddings = self.transform_matrix(sentence_embedding) # CHECK! size [batch_size, n_layer * n_emb * 2]
        # print("transformed_embeddings: " + str(transformed_embeddings.shape)) 
        transformed_embeddings = transformed_embeddings.reshape([batch_size, self.gpt2_config.n_layer, 2, self.gpt2_config.n_head, 1, int(self.gpt2_config.n_embd/self.gpt2_config.n_head)])
        # print("transformed_embeddings: " + str(transformed_embeddings.shape)) 
        transformed_embeddings = torch.transpose(transformed_embeddings,0,1).contiguous()
        transformed_embeddings = torch.transpose(transformed_embeddings,1,2).contiguous()
        # print("transformed_embeddings: " + str(transformed_embeddings.shape)) 
        

        # decoder
        if prompting_text is None:
            decoder_input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids("<|sos|>")]*batch_size, device = device).long().reshape(batch_size,1)
        else:
            prompting_text_tokens = "<|sos|>" + " " + prompting_text
            prompting_text_encoded = self.tokenizer.encode(prompting_text_tokens)
            decoder_input_ids = torch.tensor(prompting_text_encoded*batch_size, device = device).long().reshape(batch_size,len(prompting_text_encoded))
        past = transformed_embeddings


        # generate tokens
        generated = decoder_input_ids
        for _ in range(args.generate_length):

            # # DEBUGGING
            # logger.info("generated: " + str(generated))
            
            # decoder forward pass
            decoder_lm_logits, decoder_presents, decoder_hidden_states, decoder_attentions = self.decoder(input_ids = generated, past = past, attention_mask = None)


            # sample from vocabulary
            decoder_lm_logits = decoder_lm_logits[:,-1,:]

            # print("decoder_lm_logits: ")
            # print(decoder_lm_logits.shape)
            # print(decoder_lm_logits.shape[-1)
            # print(decoder_lm_logits[:3])

            filtered_decoder_lm_logits = top_k_top_p_filtering(decoder_lm_logits, top_k=args.top_k, top_p=args.top_p)
            if args.temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_decoder_lm_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_decoder_lm_logits, dim=-1), num_samples=1)                
            generated = torch.cat((generated, next_token), dim=1)
    
            ## DEBUGGING
            # if generated.shape[1]==5:
            #    logger.info("decoder_lm_logits: " + str(decoder_lm_logits))
            #    logger.info("max(decoder_lm_logits): " + str(torch.max(decoder_lm_logits[0])))
    
        return generated, decoder_attentions


    def save_pretrained(self, args, output_dir, loss_reports):

        # set up output_dir to save sub-models
        output_dir_decoder = output_dir + "/decoder/"
        output_dir_tokenizer = output_dir + "/tokenizer/"
        output_dir_transform_matrix = output_dir + "/transform_matrix/"
        if not os.path.exists(output_dir_decoder):
            os.makedirs(output_dir_decoder)            
        if not os.path.exists(output_dir_tokenizer):
            os.makedirs(output_dir_tokenizer)
        if not os.path.exists(output_dir_transform_matrix):
            os.makedirs(output_dir_transform_matrix)
        output_dir_transform_matrix = output_dir_transform_matrix + "/transform_matrix.weights"    

        # save model
        self.decoder.save_pretrained(output_dir_decoder)
        self.tokenizer.save_pretrained(output_dir_tokenizer)
        torch.save(self.transform_matrix.state_dict(),output_dir_transform_matrix)       

        # save training args and loss record
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
        loss_reports_file = open(output_dir + "/loss_reports.pkl", "wb")
        pickle.dump(loss_reports, loss_reports_file)
        
        return


    def from_pretrained(self, args):
        
        # if from checkpoint, change dir to get model
        if args.from_checkpoint:
            model_dir = args.output_dir + "/checkpoint-{}".format(str(args.start_step))
        else:
            model_dir = args.output_dir

        # loading from pre-trained
        decoder_path = model_dir + "/decoder/"
        tokenizer_path = model_dir + "/tokenizer/"
        transform_matrix_path = model_dir + "/transform_matrix/transform_matrix.weights"
        logger.info("gpt2_config: " + str(self.gpt2_config))
        self.gpt2_config.vocab_size = self.gpt2_config.vocab_size + 2 
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_path, from_tf=bool('.ckpt' in decoder_path), config=self.gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path, do_lower_case=args.do_lower_case)
        if torch.cuda.is_available():
            print("True")
            self.transform_matrix.load_state_dict(torch.load(transform_matrix_path))
        else:
            print("False")
            self.transform_matrix.load_state_dict(torch.load(transform_matrix_path, map_location=torch.device('cpu')))

        # set up for evaluating
        self.decoder.eval()
        self.transform_matrix.eval()

        # load training args
        training_args = torch.load(os.path.join(model_dir, 'training_args.bin'))
        logger.info("training_args: " + str(training_args))

        return 

# ===================== other methods ===================== #
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



