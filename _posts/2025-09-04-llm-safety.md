---
layout: post
title: Improving Safety in Large Language Models
date: 2025-09-03 14:02:00-0200
description: This blogpost explores the challenges we face at meditron and how we can improve safety in large language models.
tags: LLM LLM-safety DPO training
categories: AI
related_posts: false
related_publications: true
---

Following the recent release of [Apertus-8B](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509) and [Apertus-70B](https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509) we (*at meditron*) tried to perform a finetuning of this models using our own medical dataset. We managed to significantly improve the performance of the model on medical tasks (though still very far from the SOTA).

<!-- TBD: Performances -->

However we soon realized that this finetuning broke the model's safety, and very simple prompt could lead to dangerous outputs. This post is about the different approaches we took to improve safety in large language models.

## Generation of the dataset

We wanted to generate a dataset that we could use to finetune our model to avoid harmful and hateful content. We first *downloaded a dataset of problematic prompts* and use it to retrieve the response from our LLM.

Here is the script we use to generate *prompts* from dataset over huggingface:
```python
from datasets import load_dataset
import tqdm
import json

SOURCES = [
    ("LLM-LAT/harmful-dataset", "train", "prompt"),
    ("onepaneai/harmful-prompts", "train", "prompt"),
]

# Load the datasets and generate a jsonl file containing all prompts
f = open("prompts.jsonl", "w")
d = set()

for source, split, field in SOURCES:
    print("Processing {} - {}".format(source, split))
    dataset = load_dataset(source, split=split)
    for entry in tqdm.tqdm(dataset):
        prompt = entry[field]

        # Prevent duplicates
        prompt_key = prompt.strip().lower()
        if prompt_key in d:
            continue
        d.add(prompt_key)

        f.write(json.dumps({"prompt": prompt}) + "\n")
```

<br>

With this dataset, we then feed it to our model to generate responses. Those were then filtered by a [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model. This filtering only flags responses to be harmful or hateful,
and does not attempt any correction. We then use a method called *DPO* (*Direct Preference Optimization*) {% cite rafailov2024directpreferenceoptimizationlanguage %} to train the model to avoid such response. This method make use of a **prompt**, a **positive** sample (that the model should answer) and a **negative** sample (that the model should avoid). The sample that were flagged dangerous become **negative samples**. A placeholder sentence is then used as a **positive sample**. Crutially we also used general level prompt to show examples of prompt the model **should answer**.

Here is the script we use to generate *responses* from the model:
```python
import json
import sys
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
MODEL = "/path/to/model"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype="auto")

# For each prompt
f = open("prompts.jsonl", "r")
fo = open(f"answers-{offset}-{modulo}.jsonl", "w")

for i, line in enumerate(tqdm.tqdm(list(f))):
    prompt = json.loads(line)["prompt"]

    inputs = tokenizer.apply_chat_template([
        {
            "role": "user",
            "content": prompt
        }
    ], return_tensors="pt", tokenizer=True, add_generation_prompt=True).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, temperature=0.4, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    fo.write(json.dumps({
        "prompt": prompt,
        "answer": answer
    }) + "\n")
```

We then choosed an hardcoded sentence for negative samples, alongside a internal dataset for example of prompts the model **should answer**. We then launched a DPO training.
