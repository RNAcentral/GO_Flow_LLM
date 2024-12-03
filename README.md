## miRNA-Curator-bot

An LLM driven bot to follow the miRNA cuation flowchart developed by the UCL curation group.


## Steps

### 0
The lowest level problem is to decide whether to curate an article or not. This is something which, for now, we need the LLM to decide for us.

I have extracted the abstracts from 50 papers which Ruth's group curated, and fed them to Claude to develop a prompt to select them from a 
collection of abstracts. I tested this locally with a small model (phi3.1-medium (14B)).

This actually works surprisingly well. There are a total of 48 abstracts (I have 50 papers, but 2 don't seem to have a proper abstract, 
or just _are_ an abstract). I would expect thoe 48 to be classified as curateable. This small-ish LLM classified 46/48 (96%) as curateable
and the two false negatives do look ambiguous from a quick look. This is, I think pretty good. Running this step on 48 abstracts took 1 hour
on an M1 chip

Longer term, I think it would be good to generate some synthetic data and finetune a smaller model to do this step, as I did for abstracts
in pombase's canto system.

### 1