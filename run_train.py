# import library
import pandas as pd 
import numpy as np
import argparse
import subprocess
import os
import sys

# main function
def main():
    
    ### input arguments
    print("Reading arguments.")
    parser = argparse.ArgumentParser()
    # important arguments
    parser.add_argument("--messages_csv", default=None, type=str, required=True,
                        help="Input file containing messages of participants, having 3 columns: user_id, message_id, message.")
    parser.add_argument("--variables_csv", default=None, type=str, required=True,
                        help="Input file containing psychological variables scores of participants, having 2 or more columns: user_id, variable1, variable2, ...")
    parser.add_argument("--stage", default=None, type=str, required=True,
                        help="Set to 'process_data' to create training dataset for PsychGenerator, then set to 'train_psychgenerator' to train PsychGenerator.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, required=False,
                        help="Total number of training epochs to perform.")
    # other training arguments
    parser.add_argument("--block_size", default=64, type=int, required=False,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).")          
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, required=False,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int, required=False,
                        help="Batch size per GPU/CPU for evaluation.")    
    parser.add_argument("--learning_rate", default=5e-5, type=float, required=False,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--save_steps', type=int, default=1000, required=False,
                        help="Print every X updates steps.")
    parser.add_argument('--logging_steps', type=int, default=100, required=False,
                        help="Log every X updates steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true', required=False,
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true', required=False,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help="Random seed for initialization")
    args = parser.parse_args()

    ### change directory to this file's directory
    # print(os.getcwd())
    # os.chdir(os.path.dirname(sys.argv[0]))
    # print(os.getcwd())


    ### extract important informations
    messages_df = pd.read_csv(args.messages_csv, header = 0)
    variables_df = pd.read_csv(args.variables_csv, header = 0)
    variables_list = variables_df.columns[1:]
    n_variables = len(variables_list)

    #### run commands 
    if args.stage == "process_data":
        #################### STAGE 1: Creating the training dataset by inferring "estimated" message-level score for message
        ########## Stage 1.1: Training user-level model
        print("Running processing data.")
        ## Importing csv files to mySQL using tools from DLATK library
        # print("========= STEP: Import csv files into MySQL =========")
        # result = subprocess.run("python2 ./src/tools/csv2mySQL.py {} fb20 messages_tc \'(user_id varchar(50), message_id varchar(50), message text)\' 1".format(args.messages_csv), shell=True)
        # command = ("python2 ./src/tools/csv2mySQL.py {} fb20 messages_tc_dummy \'(user_id varchar(50), message_id varchar(50)" + "".join([", {} double default 0".format(item) for item in variables_list]) + ")\' 1").format(args.messages_csv)
        # result = subprocess.run(command, shell=True)
        # command = ("python2 ./src/tools/csv2mySQL.py {} fb20 variables_tc \'(user_id varchar(50)" + "".join([", {} double".format(item) for item in variables_list]) + ")\' 1").format(args.variables_csv)
        # result = subprocess.run(command, shell=True)
        ## Extract 1to3gram features at user-level using DLATK library commands
        # print("========= STEP: Extract 1to3 gram user-level features =========")
        # result = subprocess.run("python3 ./dlatk/dlatkInterface.py -d fb20 -t messages_tc -c user_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram", shell=True)
        # ## Train model
        # print("========= STEP: Train user-level model =========")
        # if not os.path.exists("./dlatk_trained_models"):
        #     os.makedirs("./dlatk_trained_models")
        # command = "python3 ./dlatk/dlatkInterface.py -d fb20 -t messages_tc -c user_id -f  \
        # 'feat$1to3gram$messages_tc$user_id' --train_regression --model ridgehighcv  --outcome_table variables_tc \
        # --outcomes {} --group_freq_thresh 1 --save_models \
        # --picklefile ./dlatk_trained_models/user_level_model.pickle".format(" ".join(variables_list))
        # result = subprocess.run(command, shell=True)
        # ########## Stage 1.2: Using user-level trained model to assign "estimated" message-level score for each message
        # print("========= STEP: Extract 1to3 gram message-level features =========")
        # ## Extract 1to3gram features at message-level using DLATK library commands
        # result = subprocess.run("python3 ./dlatk/dlatkInterface.py -d fb20 -t messages_tc -c message_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram", shell=True)
        # print("========= STEP: Use user-level trained model to estimate scores for messages =========")
        ## Use model to predict "estimated" score for message-level
        # command = "python3 ./dlatk/dlatkInterface.py -d fb20 -t messages_tc -c message_id --group_freq_thresh 0 --outcome_table messages_tc_dummy \
        # -f 'feat$1to3gram$messages_tc$message_id' --outcomes {} --predict_regression_to_feat estimated_scores \
        # --model ridgecv --load --picklefile ./dlatk_trained_models/user_level_model.pickle --keep_low_variance".format(" ".join(variables_list))
        # result = subprocess.run(command, shell=True)
        # print("========= STEP: Print results to csv files and process to create psychgenerator training/validating data =========")
        # ## Print to csv
        # if not os.path.exists("./processed_data"):
        #     os.makedirs("./processed_data")
        # result = subprocess.run("python3 ./dlatk/dlatkInterface.py -d fb20 -t messages_tc -c message_id -f 'feat$p_ridg_estimated_scores$messages_tc$message_id' --print_csv ./processed_data/estimated_scores.csv  --group_freq_thresh  0", shell=True)
        # ## Process the data into train and validation files
        result = subprocess.run("python3 ./src/create_train_valid_dataset.py --messages_csv ./data/messages.csv --estimated_scores_csv ./processed_data/estimated_scores.csv --output ./processed_data", shell=True)


    elif args.stage == "train_psychgenerator":
        #################### STAGE 2: Training PsyGenerator
        command = "python ./src/train_psychgenerator.py \
          --train_data_file ./processed_data/estimated_scores_train.csv \
          --eval_data_file ./processed_data/estimated_scores_valid.csv \
          --output_dir ./psychgenerator_trained_models \
          --gpt2_model_type gpt2 \
          --gpt2_model_name_or_path gpt2 \
          --do_train \
          --evaluate_during_training \
          --do_lower_case \
          --latent_size {} \
          --block_size {} \
          --per_gpu_train_batch_size {} \
          --per_gpu_eval_batch_size {} \
          --learning_rate {} \
          --save_steps {} \
          --logging_steps {} \
          --num_train_epochs {}".format(str(n_variables), str(args.block_size), str(args.per_gpu_train_batch_size), str(args.per_gpu_eval_batch_size), str(args.learning_rate), str(args.save_steps), str(args.logging_steps), str(args.num_train_epochs))
        if args.overwrite_output_dir:
            command += " --overwrite_output_dir"
        if args.overwrite_cache:
            command += " --overwrite_cache"
        result = subprocess.run(command, shell=True)

if __name__ == "__main__":
    main() 
