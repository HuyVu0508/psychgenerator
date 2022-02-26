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


    ### extract important informations
    messages_df = pd.read_csv(args.messages_csv, header = 0)
    scores_df = pd.read_csv(args.scores_csv, header = 0)
    variables_list = scores_df.columns[1:]
    n_variables = len(variables_list)

    #### run commands 
    if args.stage == "process_data":
        #################### STAGE 1: Creating the training dataset by inferring "estimated" message-level score for message
        ########## Stage 1.1: Training user-level model
        ## Importing csv files to mySQL using tools from DLATK library
        result = subprocess.run("./dlatk/tools/importmethods.py -d fb20 -t messages_tc --csv_to_mysql --csv_file {} --ignore_lines 1".format(args.messages_csv), capture_output=True, text=True)
        print("stdout:", result.stdout)
        result = subprocess.run("./dlatk/tools/importmethods.py -d fb20 -t variables_tc --csv_to_mysql --csv_file {} --ignore_lines 1".format(args.scores_csv), capture_output=True, text=True)
        print("stdout:", result.stdout)
        ## Extract 1to3gram features at user-level using DLATK library commands
        result = subprocess.run("python3 ./dlatkInterface.py -d fb20 -t messages_tc -c user_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram", capture_output=True, text=True)
        print("stdout:", result.stdout)
        ## Train model
        command = "python3 ./dlatkInterface.py -d fb20 -t messages_tc -c user_id -f  \
        'feat$1to3gram$messages_tc$user_id$16to16' --train_regression --model ridgehighcv  --outcome_table variables_tc \
        --outcomes score --feature_selection pca  --group_freq_thresh 1 --save_models \
        --picklefile ./dlatk_trained_models/user_level_model.pickle".format(" ".join(variables_list))
        result = subprocess.run(command, capture_output=True, text=True)
        print("stdout:", result.stdout)
        ########## Stage 1.2: Using user-level trained model to assign "estimated" message-level score for each message
        ## Extract 1to3gram features at message-level using DLATK library commands
        result = subprocess.run("python3 ./dlatkInterface.py -d fb20 -t messages_tc -c message_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram", capture_output=True, text=True)
        print("stdout:", result.stdout)
        ## Use model to predict "estimated" score for message-level
        command = "python3 ./dlatkInterface.py -d fb20 -t messages_tc -c message_id --group_freq_thresh 0 --outcome_table variables_tc \
        -f 'feat$1to3gram$messages_tc$message_id$16to16' --outcomes {} --predict_regression_to_feat estimated_scores \
        --model ridgecv --load --picklefile ./dlatk_trained_models/user_level_model.pickle --keep_low_variance".format(" ".join(variables_list))
        result = subprocess.run(command, capture_output=True, text=True)
        print("stdout:", result.stdout)
        ## Print to csv
        result = subprocess.run("python3 ./dlatkInterface.py -d fb20 -t messages_tc -c message_id -f 'feat$p_ridg_estimated_scores$messages_tc$message_id' --print_csv estimated_scores.csv  --group_freq_thresh  0", capture_output=True, text=True)
        print("stdout:", result.stdout)
        ## Process the data into train and validation files
        result = subprocess.run("python3 ./create_train_valid_dataset.py --input estimated_scores.csv --output_prefix estimated_scores", capture_output=True, text=True)
        print("stdout:", result.stdout)



    # #### run commands 
    # if args.stage == "process_data":
    #     #################### STAGE 1: Creating the training dataset by inferring "estimated" message-level score for message
    #     ########## Stage 1.1: Training user-level model
    #     ## Importing csv files to mySQL using tools from DLATK library
    #     result = subprocess.run("./dlatk/tools/importmethods.py -d database -t messages --csv_to_mysql --csv_file {} --ignore_lines 1".format(args.messages_csv), capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     result = subprocess.run("./dlatk/tools/importmethods.py -d database -t variables --csv_to_mysql --csv_file {} --ignore_lines 1".format(args.scores_csv), capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     ## Extract 1to3gram features at user-level using DLATK library commands
    #     result = subprocess.run("python3 ~/dlatk/dlatkInterface.py -d database -t messages -c user_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram", capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     ## Train model
    #     command = "python3 ~/dlatk/dlatkInterface.py -d database -t messages -c user_id -f  \
    #     'feat$1to3gram$messages$user_id$16to16' --train_regression --model ridgehighcv  --outcome_table variables \
    #     --outcomes score --feature_selection pca  --group_freq_thresh 1 --save_models \
    #     --picklefile ./dlatk_trained_models/user_level_model.pickle".format(" ".join(variables_list))
    #     result = subprocess.run(command, capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     ########## Stage 1.2: Using user-level trained model to assign "estimated" message-level score for each message
    #     ## Extract 1to3gram features at message-level using DLATK library commands
    #     result = subprocess.run("python3 ~/dlatk/dlatkInterface.py -d database -t messages -c message_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram", capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     ## Use model to predict "estimated" score for message-level
    #     command = "python3 ~/dlatk/dlatkInterface.py -d database -t messages -c message_id --group_freq_thresh 0 --outcome_table variables \
    #     -f 'feat$1to3gram$messages$message_id$16to16' --outcomes {} --predict_regression_to_feat estimated_scores \
    #     --model ridgecv --load --picklefile ./dlatk_trained_models/user_level_model.pickle --keep_low_variance".format(" ".join(variables_list))
    #     result = subprocess.run(command, capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     ## Print to csv
    #     result = subprocess.run("python3 ~/dlatk/dlatkInterface.py -d database -t messages -c message_id -f 'feat$p_ridg_estimated_scores$messages$message_id' --print_csv estimated_scores.csv  --group_freq_thresh  0", capture_output=True, text=True)
    #     print("stdout:", result.stdout)
    #     ## Process the data into train and validation files
    #     result = subprocess.run("python3 ./create_train_valid_dataset.py --input estimated_scores.csv --output_prefix estimated_scores", capture_output=True, text=True)
    #     print("stdout:", result.stdout)




    # elif args.stage == "train_psychgenerator":
    #     #################### STAGE 2: Training PsyGenerator
    #     command = "python train_psychgenerator.py \
    #       --train_data_file ./data/estimated_scores_train.csv \
    #       --eval_data_file ./data/estimated_scores_train.csv \
    #       --output_dir ./trained_models \
    #       --gpt2_model_type gpt2 \
    #       --gpt2_model_name_or_path gpt2 \
    #       --do_train \
    #       --evaluate_during_training \
    #       --do_lower_case \
    #       --latent_size {} \
    #       --block_size {} \
    #       --per_gpu_train_batch_size {} \
    #       --per_gpu_eval_batch_size {} \
    #       --learning_rate {} \
    #       --save_steps {} \
    #       --logging_steps {} \
    #       --num_train_epochs {}".format(str(n_variables), str(args.block_size), str(args.per_gpu_train_batch_size), str(args.per_gpu_eval_batch_size), str(args.learning_rate), str(args.save_steps), str(args.logging_steps), str(args.num_train_epochs))
    #     if args.overwrite_output_dir:
    #         command += " --overwrite_output_dir"
    #     if args.overwrite_cache:
    #         command += " --overwrite_cache"
    #     result = subprocess.run(command, capture_output=True, text=True)
    #     print("stdout:", result.stdout)