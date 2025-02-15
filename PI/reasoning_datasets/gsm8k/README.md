---
annotations_creators:
- crowdsourced
language_creators:
- crowdsourced
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- text2text-generation
task_ids: []
paperswithcode_id: gsm8k
pretty_name: Grade School Math 8K
tags:
- math-word-problems
dataset_info:
- config_name: main
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 3963202
    num_examples: 7473
  - name: test
    num_bytes: 713732
    num_examples: 1319
  download_size: 2725633
  dataset_size: 4676934
- config_name: socratic
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 5198108
    num_examples: 7473
  - name: test
    num_bytes: 936859
    num_examples: 1319
  download_size: 3164254
  dataset_size: 6134967
configs:
- config_name: main
  data_files:
  - split: train
    path: main/train-*
  - split: test
    path: main/test-*
- config_name: socratic
  data_files:
  - split: train
    path: socratic/train-*
  - split: test
    path: socratic/test-*
---

# Dataset Card for GSM8K

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-instances)
  - [Data Splits](#data-instances)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** https://openai.com/blog/grade-school-math/
- **Repository:** https://github.com/openai/grade-school-math
- **Paper:** https://arxiv.org/abs/2110.14168
- **Leaderboard:** [Needs More Information]
- **Point of Contact:** [Needs More Information]

### Dataset Summary

GSM8K (Grade School Math 8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The dataset was created to support the task of question answering on basic mathematical problems that require multi-step reasoning.
- These problems take between 2 and 8 steps to solve.
- Solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.
- A bright middle school student should be able to solve every problem: from the paper, "Problems require no concepts beyond the level of early Algebra, and the vast majority of problems can be solved without explicitly defining a variable."
- Solutions are provided in natural language, as opposed to pure math expressions. From the paper: "We believe this is the most generally useful data format, and we expect it to shed light on the properties of large language models’ internal monologues""

### Supported Tasks and Leaderboards

This dataset is generally used to test logic and math in language modelling.
It has been used for many benchmarks, including the [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

### Languages

The text in the dataset is in English. The associated BCP-47 code is `en`.

## Dataset Structure

### Data Instances

For the `main` configuration, each instance contains a string for the grade-school level math question and a string for the corresponding answer with multiple steps of reasoning and calculator annotations (explained [here](https://github.com/openai/grade-school-math#calculation-annotations)).


```python
{
    'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
    'answer': 'Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72',
}
```

For the `socratic` configuration, each instance contains a string for a grade-school level math question, a string for the corresponding answer with multiple steps of reasoning, calculator annotations (explained [here](https://github.com/openai/grade-school-math#calculation-annotations)), and *Socratic sub-questions*.

```python
{
    'question': 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?',
    'answer': 'How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72',
}
```

### Data Fields

The data fields are the same among `main` and `socratic` configurations and their individual splits.

- question: The question string to a grade school math problem.

- answer: The full solution string to the `question`. It contains multiple steps of reasoning with calculator annotations and the final numeric solution.

### Data Splits

| name   |train|validation|
|--------|----:|---------:|
|main    | 7473|      1319|
|socratic| 7473|      1319|

## Dataset Creation

### Curation Rationale

[Needs More Information]

### Source Data

#### Initial Data Collection and Normalization

From the paper, appendix A:

> We initially collected a starting set of a thousand problems and natural language solutions by hiring freelance contractors on Upwork (upwork.com). We then worked with Surge AI (surgehq.ai), an NLP data labeling platform, to scale up our data collection. After collecting the full dataset, we asked workers to re-solve all problems, with no workers re-solving problems they originally wrote. We checked whether their final answers agreed with the original solutions, and any problems that produced disagreements were either repaired or discarded. We then performed another round of agreement checks on a smaller subset of problems, finding that 1.7% of problems still produce disagreements among contractors. We estimate this to be the fraction of problems that contain breaking errors or ambiguities. It is possible that a larger percentage of problems contain subtle errors.

#### Who are the source language producers?

[Needs More Information]

### Annotations

#### Annotation process

[Needs More Information]

#### Who are the annotators?

Surge AI (surgehq.ai)

### Personal and Sensitive Information

[Needs More Information]

## Considerations for Using the Data

### Social Impact of Dataset

[Needs More Information]

### Discussion of Biases

[Needs More Information]

### Other Known Limitations

[Needs More Information]

## Additional Information

### Dataset Curators

[Needs More Information]

### Licensing Information

The GSM8K dataset is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Citation Information

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

### Contributions

Thanks to [@jon-tow](https://github.com/jon-tow) for adding this dataset.