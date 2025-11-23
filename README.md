# NeuralNet Project

This project has 3 parts:
1. Loading a dataset and training 
2. Backpropagation (BP) neural network implementation
3. Obtain and compare predictions using the three models (BP, BP-F, MLR-F)

## Project Structure

```
NeuralNet/
│
├── data/                # Dataset files (CSV, TXT, etc.)
│
├── src/                 # Source code
│   ├── NeuronalDataset.py    # Code to load and preprocess the dataset
│   └── neuralnet    # Student's BP neural network implementation
│
├── notebooks/           # Jupyter notebooks for experiments and analysis
│   ├── optional/     # Optional activities
│   ├── 01_preprocessing.ipynb     # Code to load and preprocess the dataset (Part 1)
│   └── 02_BP_from_scratch.ipynb    # BP neural network implementation (Part 2)
│
├── test /               # Unit tests for code modules
│
├── report /               # Reports generated for code results
│
├── README.md            # Project documentation
│
└── requirements.txt     # Python dependencies
```

## Getting Started

1. Clone the repository.
2. Install dependencies:  
    ```
    pip install -r requirements.txt
    ```
3. Place your dataset in the `data/` directory.
4. Load and training script (Part 1):  
    ```
    python src/NeuronalDataset.py
    ```

5. Implementation of BP (Part 2):  
    ```
    python src/neuralnet
    ```


## Features
- Dataset loading and preprocessing
- Training and evaluation scripts
- Custom BP neural network implementation
- Example notebooks

## Author
Victoria Joven
