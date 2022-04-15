# PsychGenerator: Artificially Intelligent Language with Personality

This is the code base for the PsychGenerator, a language generative model with the control of underlying psychology constructs, with additional demographics.

## Codebase overview
Below is the descriptions of files in the repository.
*`./data/messages.csv`: file containing three columns "user_id", "message_id" and "message", indicates users and their texts messages.
*`./data/variables.csv`: file containing two columns "user_id", "variable1", "variable2", "variable3", indicates the psychological traits score value for each user
*`./src`: folder containing source codes for training model and generating.
*`./run_train.py`: Python interface to run training model.
*`./run_generate.py`: Python interface to run generating texts from trained model.

## Installations requirements
In order to run our code, the following Python libraries and other dependents are required.
* Python library transformers: https://huggingface.co/docs/transformers/index
* Python library DLATK: https://dlatk.wwbp.org/index.html
* MySQL database management system: https://dev.mysql.com 

## Instructions for training and using PsychGenerator

### Training PsychGenerator
To train our model, the first step is preocessing and creating training data, which is "estimated" scores for each message.
```
   python3 ./run_train.py \
	--stage process_data \
	--messages_csv ./data/messages.csv \
	--variables_csv ./data/variables.csv
```
After obtaining training data, we train PsychGenerator by running the following command. There are many configurations for the training process that can be modifed (e.g., number of epochs, learning rates). Run `python3 ./run_train.py -h` for more information.
```
   python3 ./run_train.py \
	--stage train_psychgenerator \
	--messages_csv ./data/messages.csv \
	--variables_csv ./data/variables.csv \
	--num_train_epochs 5
```

### Gnerating texts with PsychGenerator
After training, PsychGenerator can be used to generate texts correspond to all interested dimensions, using the following commands.
```
   python3 ./run_generate.py \
	--messages_csv ./data/messages.csv \
	--variables_csv ./data/variables.csv \
	--output_dir "./trained_models" \
	--generate_num 5 \
	--prompting_text "I like to"
```

In order to control for demograhics (e.g., age, gender). Add switches `demographics_variable` and `k_value_demographics` to the command as below.
```
   python3 ./run_generate.py \
	--output_dir "./trained_models" \
	--generate_num 5 \
	--prompting_text "I like to" \
	--k_value 3 \
	--demographics_variable variable3 \
	--k_value_demographics 3 \
	--seed 42 
```