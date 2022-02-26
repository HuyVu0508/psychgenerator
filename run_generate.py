# import library
import pandas as pd 
import numpy as np
import argparse
import subprocess

# main function
def main():
    
    ### input arguments
    print("Reading arguments.")
    parser = argparse.ArgumentParser()
    # important arguments
    parser.add_argument("--messages_csv", default=None, type=str, required=True,
                        help="Input file containing messages of participants, having 3 columns: user_id, message_id, message.")
    parser.add_argument("--scores_csv", default=None, type=str, required=True,
                        help="Input file containing psychological scores of participants, having 2 or more columns: user_id, score1, score2, ...")
    # other training arguments
    parser.add_argument("--block_size", default=64, type=int, required=Fase,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
    parser.add_argument("--generate_num", type=10, default=None, required=Fase,
                        help="Number of messages generated at each position."
    parser.add_argument("--generate_length", type=64, default=None, required=Fase,
                        help="Maximum length of generated texts."
    parser.add_argument("--temperature", type=float, default=1.0, required=Fase,
                        help="Temperature of nucleus sampling."
    parser.add_argument("--top_k", type=int, default=10, required=Fase,
                        help="Top-k value of nucleus sampling."
    parser.add_argument("--top_p", type=float, default=0.9, required=Fase,
                        help="Top-p value of nucleus sampling."
    parser.add_argument("--prompting_text", type=str, default="I like to", required=Fase,
                        help="Initial prompting texts to start generating from."
    parser.add_argument('--k_value', type=int, default=3, required=Fase,
                        help="Position value to generate texts from."
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization.")

    args = parser.parse_args()


    # extract important informations
    messages_df = pd.read_csv(args.messages_csv, header = 0)
    scores_df = pd.read_csv(args.scores_csv, header = 0)
    variables_list = scores_df.columns[1:]

    # run commmand
    command = "python inference_psychgenerator.py \
        --train_data_file ./data/estimated_scores_train.csv \
        --output_dir ./trained_models \
        --gpt2_model_type gpt2 \
        --gpt2_model_name_or_path gpt2 \
        --latent_size {} \
        --block_size {} \
        --do_lower_case \
        --temperature {} \
        --top_k {} \
        --top_p {} \
        --generate_num {} \
        --generate_length {} \
        --prompting_text {} \
        --k_value {} \
        --seed {}".format(str(n_variables), str(args.generate_num), str(args.generate_length), str(args.prompting_text), str(args.k_value))
    result = subprocess.run(command, capture_output=True, text=True)
    print("stdout:", result.stdout)    