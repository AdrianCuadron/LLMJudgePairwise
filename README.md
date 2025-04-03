# Evaluating Natural Language Generation in the Medical Domain

## Description

This repository contains the source code and experiments described in our study on evaluating Natural Language Generation (NLG) in the medical domain. Our approach implements a pairwise LLM-judge evaluation method, following the Ranking Over Scoring framework.

Evaluating NLG in the medical domain is challenging due to data scarcity and the limitations of traditional reference-based metrics like BLEU and ROUGE. Recent advancements in Large Language Models (LLMs) have led to the adoption of LLM-as-judges for evaluation, but this approach introduces biases such as:
- Self-enhancement bias
- Positional bias
- Verbosity bias

To address these issues, we implement a pairwise evaluation methodology using open and cost-free models, applying it across proxy tasks such as:
- Question Answering (QA)
- Natural Language Inference (NLI)
- Misinformation Detection

Our study analyzes alignment with human preferences and the mitigation of biases in automatic evaluation of LLM-generated medical arguments.

## Content 
Python Scripts:
- llmAsJudge-pairwise.py: script to prompt the LLM-judges and get the outputs
- parse_outputs_pairwise.py: script to parse the outputs and calculate the rankings based on victory count or TrueSkill score

In 'results' we include the rankings obtained by each model.
