# FlyGenie: Minimize your flight ticket purchase costs
## AA228-CS238 Final Project Repository

[AA228/CS238: Decision Making under Uncertainty](https://aa228.stanford.edu), Winter 2023, Stanford University.

This repository contains the code/implementation for the final project for CS238.

Below are the details of the files in the repo:

    Final_project/                                      # Root repo
    ├── dataset  
    |   ├── train                                       # CSV data files that contain the web-scraped flight info
    │       ├── 13_03.csv               
    │       ├── 14_03..csv              
    |       .
    |       .
    |       └── 19_03.csv 
    |   ├── (Delhi/Bangalore/Kolkata)_mdp.csv           # MDP files generated after training
    |   ├── (Delhi/Bangalore/Kolkata)_mdp.policy        # Policy files generated from the MDP files
    |   ├── test_17_03.csv                              # File used for testing (extra data scraped on 17th March 2023)
    |   ├── test_(Delhi/Bangalore/Kolkata).csv          # Route-wise split CSV file from test_17_03.csv
    |   ├── (Delhi/Bangalore/Kolkata)_test_mdp.csv      # MDP files generated for each route from the test file
    |   ├── (Delhi/Bangalore/Kolkata)_test_mdp.policy   # Policy files generated from the test MDP files
    |   └── test_(Delhi/Bangalore/Kolkata)metrics.csv   # Route-wise split results/metrics file 
    └── qlearning.jl                                    # Code for the Q-learning algorithm to help)

## Instructions to run
Please run the following command from the root repo

#### For training over all files in dataset/train/
```
julia qlearning.jl
```

#### For evaluating performance of the structured learning algorithm (Q-learning) over a test file
```
julia qlearning.jl <test_file>
```
