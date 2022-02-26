Below are step-by-step instructions to use our code to create PsychGenerator working with other variables/dataset.

We start with two fundamental data files:
 + messages.csv: file containing three columns "user_id", "message_id" and "message", indicates users and their texts messages
 + variables.csv: file containing two columns "user_id", "score", indicates the psychological traits score value for each user

Installations requirements:
 + Python library transformers: https://huggingface.co/docs/transformers/index
 + Python library DLATK: https://dlatk.wwbp.org/index.html
 
Code base descriptions:
 + src: containing source codes for training model and generating
 + run_train.py: used to train model
 + run_generate.py: used to generate texxts
 

Instructions for training and using PsychGenerator:
 + Step 1: training model
   Command:
   # create training data
   python3 run_train.py --stage process_data --messages_csv messages.csv --scores_csv variables.csv
   # train PsychGenerator used the processed training data above
   python3 run_train.py --stage train_psychgenerator --messages_csv messages.csv --scores_csv variables.csv --num_train_epochs 5


 + Step 2: generating texts with trained model
   Command:
   python3 run_generate.py --output_dir "./trained_models" --generate_num 5 --prompting_text "I like to"

