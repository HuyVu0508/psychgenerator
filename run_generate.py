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
    parser.add_argument("--variables_csv", default=None, type=str, required=True,
                        help="Input file containing psychological scores of participants, having 2 or more columns: user_id, variable1, variable2, ...")
    # other training arguments
    parser.add_argument('--demographics_variable', type=str, default=None, required=False,
                        help="The name of demographics variable if there is.")
    parser.add_argument('--k_value', type=float, default=3.0, required=False,
                        help="Position value to generate texts from.")
    parser.add_argument('--k_value_demographics', type=float, default=3.0, required=False,
                        help="Position value to generate texts from for demographics variable.")
    parser.add_argument("--block_size", default=64, type=int, required=False,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
    parser.add_argument("--generate_num", default=10, type=int, required=False,
                        help="Number of messages generated at each position.")
    parser.add_argument("--generate_length", default=64, type=int, required=False,
                        help="Maximum length of generated texts.")
    parser.add_argument("--temperature", type=float, default=1.0, required=False,
                        help="Temperature of nucleus sampling.")
    parser.add_argument("--top_k", type=int, default=10, required=False,
                        help="Top-k value of nucleus sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, required=False,
                        help="Top-p value of nucleus sampling.")
    parser.add_argument("--prompting_text", type=str, default="", required=False,
                        help="Initial prompting texts to start generating from.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization.")

    args = parser.parse_args()


    # extract important informations
    messages_df = pd.read_csv(args.messages_csv, header = 0)
    variables_df = pd.read_csv(args.variables_csv, header = 0)
    variables_list = variables_df.columns[1:]
    n_variables = len(variables_list)


    # run commmand
    command = "python ./src/inference_psychgenerator.py \
        --train_data_file ./processed_data/estimated_scores_train.csv \
        --output_dir ./psychgenerator_trained_models \
        --gpt2_model_type gpt2 \
        --gpt2_model_name_or_path gpt2 \
        --method {} \
        --demographics_variable {} \
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
        --k_value_demographics {} \
        --seed {}".format(("variables_demographics_inference" if args.demographics_variable is not None else "variables_inference"), str(args.demographics_variable), str(n_variables), str(args.block_size), str(args.temperature), str(args.top_k), str(args.top_p), str(args.generate_num), str(args.generate_length), str("\"" + args.prompting_text + "\""), str(args.k_value), str(args.k_value_demographics),str(args.seed))
    result = subprocess.run(command, shell=True)

if __name__ == "__main__":
    main() 
