import os
from typing import Literal

import pandas as pd

choices = ["A", "B", "C", "D"]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx):
    prompt = df.iloc[idx, 0]

    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    # prompt += "\nAnswer:"
    answer = df.iloc[idx, k + 1]
    wrong_answer = [choice for choice in choices if choice != answer]
    return prompt, answer, wrong_answer

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def gen_mmlu_datasets(data_dir, phase: Literal["test", "dev", "val"]): 
    dataset = []
    subjects = sorted([f.split(f"_{phase}.csv")[0] for f in os.listdir(os.path.join(data_dir, f"{phase}")) if f"_{phase}.csv" in f])
    for subject in subjects: 
        data_df = pd.read_csv(os.path.join(data_dir, f"{phase}", subject + f"_{phase}.csv"), header=None)
        for i in range(data_df.shape[0]): 
            prompt, answer, wrong_answer = format_example(data_df, i)
            dataset.append((prompt, answer, wrong_answer))
    return dataset