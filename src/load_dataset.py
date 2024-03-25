import pandas as pd
import datasets
import numpy as np
import logging
import sys

sys.path.insert(0, '../Basque_EDA')

from augment import gen_eda_df

def load_dataset(runArgs, trainingArgs, tokenizer):

    """

    This function load the dataset, apply EDA if needed and tokenize the data

    Args:

        runArgs (RunArgs): RunArgs object containing the arguments to run the model
        trainingArgs (TrainingArguments): TrainingArguments object containing the arguments to train the model
        tokenizer (AutoTokenizer): Tokenizer to use

    Returns:

        tuple: Tuple containing the train, eval and test datasets
        
    """

    # Load datasets from csv files
    splits = {k:pd.read_csv(f"{runArgs.data_path}/{k}.csv") for k in ["train", "eval", "test"]}

    if runArgs.use_eda:

        alpha = runArgs.eda_alpha
        num_aug = runArgs.eda_n

        splits["train"] = gen_eda_df(splits["train"], alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, alpha_rd=alpha, num_aug=num_aug)

    # Tokenize datasets
    for name, split in splits.items():

        logging.info(f"Split: {name} - {len(splits[name])} samples. Values percentages {splits[name]['label'].value_counts(normalize=True)}")

        # Load data as datasets object
        k_split = datasets.Dataset.from_pandas(
            split,
            split=name,
        )

        # Tokenize data
        splits[name] = k_split.map(
            tokenize_data,
            batched=True,
            desc=f"Running tokenizer on {name} dataset",
            fn_kwargs={"tokenizer":tokenizer, "runArgs":runArgs},
        )

        splits[name] = splits[name].shuffle(seed=trainingArgs.seed)

    return splits.values()

def tokenize_data(examples, tokenizer, runArgs):

    label2id = runArgs.label2id

    tokenized_inputs = tokenizer(examples["text"], truncation=True,
                                padding="max_length", max_length = runArgs.max_length, return_tensors="np")

    tokenized_inputs["label"] = np.array(list(map(lambda x: label2id[x], examples["label"])))

    return tokenized_inputs