import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import random


# main function
def main():
    
    ### input arguments
    print("Reading arguments.")
    parser = argparse.ArgumentParser()
    # important arguments
    parser.add_argument("--messages_csv", default=None, type=str, required=True,
                        help="Input file containing estimated scores which is used to create train/validate files for psychgenerator.")
    parser.add_argument("--estimated_scores_csv", default=None, type=str, required=True,
                        help="Input file containing estimated scores which is used to create train/validate files for psychgenerator.")
    parser.add_argument("--output", default=None, type=str, required=True,
                        help="Folder to contain created train/validate files for psychgenerator.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()



    # input
    print("Reading files.")
    outcome_message_file = args.estimated_scores_csv
    outcome_message_df = pd.read_csv(outcome_message_file, header=0, index_col=0)
    outcome_list = list(outcome_message_df.columns[0:])
    outcome_message_df['message_id'] = outcome_message_df.index
    outcome_message_df['message_id'] = outcome_message_df['message_id'].astype(str)
    outcome_message_df = outcome_message_df[["message_id"] + outcome_list]
    outcome_message_df.columns = ['message_id'] + outcome_list
    messages_file = args.messages_csv
    messages_df = pd.read_csv(messages_file, header=0, index_col=None)
    messages_df['message_id'] = messages_df['message_id'].astype(str)
    # output
    dataset_train_path = args.output + "/estimated_scores_train.csv"
    dataset_valid_path = args.output + "/estimated_scores_valid.csv"


    # print out results
    print(outcome_message_df.head())
    print(messages_df.head())
    print("len(outcome_message_df): " + str(len(outcome_message_df)))
    print("len(messages_df): " + str(len(messages_df)))


    # processing messages
    print("Processing messages.")
    messages_df["message"] = messages_df["message"].str.lower()
    # remove all non-alphabet => check this message_id: messages_df.loc[[item.startswith("today is the day you have made, i will rejoice and") for item in messages_df['message']]].message
    ## messages_df.loc[73470]
    from string import printable
    keep_index = []
    for item in messages_df['message']:
        key = True
        for x in item:
            if ord(x)>127:
                key = False
                break
        keep_index.append(key)
    messages_df = messages_df.loc[keep_index]
    # replace line break "\n" with " "
    messages_df['message'] = [str(item).replace("\\n"," ") for item in messages_df['message']]
    # remove links http
    messages_df = messages_df.loc[[True if ('http:' not in str(item)) else False for item in messages_df['message']]]
    # remove messages shorter than
    L = 5
    messages_df = messages_df.loc[[True if len(item.strip(" "))>5 else False for item in messages_df['message']]]
    print(messages_df.head())


    # set index
    print("Setting indexes.")
    outcome_message_df.index = outcome_message_df['message_id']
    messages_df['index'] = messages_df['message_id']
    messages_df.index = messages_df['index']
    print("outcome_message_df.index: ")
    print(outcome_message_df.head())
    print("messages_df.index: ")
    print(messages_df.head())


    # filter messages_df to include only messages that are in outcome_message_df
    messages_df = messages_df.loc[[True if item in outcome_message_df.index else False for item in messages_df['message_id']]]


    # merge data
    print("Merging data.")
    dataset_df = messages_df[["index", "message_id", "message"]]
    for outcome in outcome_list:
        print(outcome_message_df.loc[dataset_df["message_id"]][outcome].values)
        dataset_df[outcome] = outcome_message_df.loc[dataset_df["message_id"]][outcome].values
    dataset_df = dataset_df.fillna(0) 
    print(dataset_df.head())


    # assign message_id as index
    outcome_message_column = outcome_list
    dataset_df = dataset_df[['index', 'message'] + outcome_message_column]
    dataset_df.columns = ['message_id', 'message'] + outcome_message_column


    # shuffle data or not
    print("Shuffling data (or not).")
    random.seed(args.seed)
    dataset_df = dataset_df.sample(frac = 1)
    print(dataset_df.head())


    # divide to train/validate
    print("Dividing to train/validate partition.")
    n_train = int(0.9*len(dataset_df))
    dataset_train_df = dataset_df[:n_train]
    dataset_valid_df = dataset_df[n_train:]   
    print("len(dataset_train_df): " + str(len(dataset_train_df)))
    print("len(dataset_valid_df): " + str(len(dataset_valid_df)))


    # save to file
    dataset_train_df.to_csv(dataset_train_path, header=True, index=False)
    dataset_valid_df.to_csv(dataset_valid_path, header=True, index=False)

if __name__ == "__main__":
    main() 
