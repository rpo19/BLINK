# Entity Linking with NIL Prediction for Incremental Knowledge Base Population
This repository is a fork of
[https://github.com/facebookresearch/BLINK](https://github.com/facebookresearch/BLINK)
and is part of my master thesis work (the thesis is available
[here](https://gitlab.com/rpo254/master-thesis)).

BLINK has been adapted to the context of Incremental Knowledge Base Population
by:
- adding a Linear Regression-based NIL prediction module downstream to BLINK;
- exploiting BLINK mentions' representation to add newly identified entities for future linkings.

This repository contains:
- the scripts required for the creation of the dataset for the NIL prediction task;
- the scripts that train and evaluate (ablation study of the features for NIL prediction) the NIL prediction models;
- the scripts that simulated the human-in-the-loop-integration;
- the script that simulated the knowledge base population;

Finally a web-based demo of "BLINK + NIL prediction" using the "BLINK KB" is available. Look [here](demo.md) for more.

# Setup
For setting up the repository and reproduce the results read [here](setup.md)

# BLINK Readme
[Here](BLINK_README.md) is the default BLINK README.