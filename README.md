# PsychGenerator: Artificially Intelligent Language with Personality
This is the source code repository for the paper "Artificially Intelligent Language with Personality". This work is currently under submission.

This work proposes the architecture Psychgenerator - an transformer-based AI language model that is able to reflect individual characteristics in its text output. PsychGenerator is trained to be able to reflect any of the Big Five personality traits (openness, conscientiousness, extraversion, agreeableness, and neuroticism) as well as mental health variables (depression and life satisfaction), while optionally being conditioned on demographics (e.g., age). The live-demo of our model can be found at: http://3.12.111.1 (not for distributing until manuscript acceptance). 

This project was done in collaboration between PhD students, postdocs, and professors from Stony Brook University (Huy Vu, Swanie Juhng, Adithya Ganesan, Oscar N.E. Kjell, H. Andrew Schwartz), Stanford University (Johannes C. Eichstaedt), New York University (Joao Sedoc), University of Melbourne (Margaret L. Kern), University of Pennsylvania (Lyle Ungar). Corresponding authors: Huy Vu (hvu@cs.stonybrook.edu), Johannes C. Eichstaedt (johannes.stanford@gmail.com), H. Andrew Schwartz (has@cs.stonybrook.edu).

## Codebase overview
Below are the descriptions of files and directories in the repository.
* `./data/messages.csv`: file containing three columns "user_id", "message_id" and "message", indicating users and their text messages.
* `./data/variables.csv`: file containing column "user_id" and variables columns, such as "variable1", "variable2", "variable3", indicating the psychological traits score and/or demographics variables value for each user.
* `./src`: directory containing source codes for model architecture, model training and inferencing.
* `./run_train.py`: Python interface to run training model.
* `./run_generate.py`: Python interface to run generating text from trained model.

## Installations requirements
Operating systems: Linux (Ubuntu 16.04), MacOS, Windows.

Python: 3.6.0+.

In order to run our code, the following Python libraries and other dependents are required.
* Python library transformers: https://huggingface.co/docs/transformers/index (2.3.0+)
* Python library DLATK: https://dlatk.wwbp.org/index.html (1.1.6+)
* MySQL database management system: https://dev.mysql.com 

## Instructions for training and generating text with PsychGenerator

### Training PsychGenerator
To train our model, the first step is pre-processing and creating training data, which is the "estimated" scores for each message. The following command execute these steps. A directory `./processed_data` will be created containing the processed training and validating data. 
```
   python3 ./run_train.py \
	--stage "process_data" \
	--messages_csv ./data/messages.csv \
	--variables_csv ./data/variables.csv
```
After obtaining the training data, we train PsychGenerator by running the following command. There are many configurations for the training process that can be modifed (e.g., number of epochs, learning rates). Run `python3 ./run_train.py -h` for more information. The code reads the processed data from `./processed_data` directory then begins the training process. A directory `./trained_models` will be created containing the trained model.
```
   python3 ./run_train.py \
	--stage "train_psychgenerator" \
	--messages_csv ./data/messages.csv \
	--variables_csv ./data/variables.csv \
	--num_train_epochs 5
```

### Generating text with PsychGenerator
After training, PsychGenerator can be used to generate text corresponding to all interested dimensions, using the following command. The code loops through all variables and generates text from the high and low value of each variable. There are many configurations for the generating process that can be modifed (e.g., number of generated sentences, nuclous sampling parameters). Run `python3 ./run_generate.py -h` for more information.
```
   python3 ./run_generate.py \
	--messages_csv ./data/messages.csv \
	--variables_csv ./data/variables.csv \
	--output_dir ./trained_models \
	--generate_num 5 \
	--prompting_text "I like to"
```

In order to control for demograhics (e.g., age, gender). Add arguments `demographics_variable` and `k_value_demographics` to the command as below. Argument `demographics_variable` indicates the name of the demographics variable. Argument `k_value_demographics` indicates the k-value from which the model will generate text from. In the example below, variable3 is the demographics variable, while variable1 and variable2 are psychological trait variabsle.
```
   python3 ./run_generate.py \
	--output_dir ./trained_models \
	--generate_num 5 \
	--prompting_text "I like to" \
	--k_value 3 \
	--demographics_variable variable3 \
	--k_value_demographics 3 \
	--seed 42 
```
