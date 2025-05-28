import os
from typing import Literal
import pandas as pd

choices = ["A", "B", "C", "D", "E"]

def format_example(df, idx):
    df = df[["question", "choices", "answerKey"]]
    prompt = df.iloc[idx, 0]
    choice_answer = df.iloc[idx, 1]

    for i in range(len(choice_answer["label"])): 
        prompt += "\n{}. {}".format(choice_answer["label"][i], choice_answer["text"][i])
    answer = df.iloc[idx, 2]
    wrong_answer = [choice for choice in choices if choice != answer]
    return prompt, answer, wrong_answer


def gen_csqa_datasets(data_dir, phase: Literal["train", "validation", "test"]): 
    dataset = []
    splits = {'train': os.path.join(data_dir, 'train-00000-of-00001.parquet'), 
              'validation': os.path.join(data_dir, 'validation-00000-of-00001.parquet'),
              'test': os.path.join(data_dir, 'test-00000-of-00001.parquet')}

    data_df = pd.read_parquet(splits[phase])
    for i in range(data_df.shape[0]): 
        prompt, answer, wrong_answer = format_example(data_df, i)
        dataset.append((prompt, answer, wrong_answer))
    return dataset