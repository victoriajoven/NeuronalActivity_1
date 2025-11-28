# NeuralNet Project

This project has 4 parts:
1. Part 1: Loading a dataset and training.  Code to load and preprocess the dataset
2. Part 2: Backpropagation (BP) neural network implementation
3. Part 3: Comparative models.  Obtain and compare predictions using the three models (BP, BP-F, MLR-F)
4. Part 4: Optional practice (part 1) - BPF regularization

## Project Structure

```
NeuralNet/
│
├── data/                # Dataset files (CSV, TXT, etc.)
│
├── src/                 # Source code
│   ├── data_generators/    # Code to generate the dataset
│   │   ├── NeuronalDataset.py        # Part 1 (Python project)
│   │   └── generatePriceCarscsv.py   # CSV dataset generator (only if you need generate new data)
│   └── neuralnet    # Student's BP neural network implementation
│       ├── __init__.py        
│       ├── BP_Multilayer.py          # Part 2 (Python project)
│       ├── model_comparison.py       # Part 3 (Python project)
│       └── util.py                   # Project utils
│
├── notebooks/           # Jupyter notebooks for experiments and analysis
│   ├── 01_preprocessing.ipynb     # Part 1 (Jupyter netbook)
│   ├── 02_BP_from_scratch.ipynb   # Part 2 (Jupyter netbook)
│   ├── 03_model_comparison.ipynb  # Part 3 (Jupyter netbook)
│   └── 04_optional_par1.ipynb     # Part 4 (Jupyter netbook) 
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

### OPTION 1: Jupyter netbooks (recommended)
```text
4. Run 01_preprocessing.ipynb to load dataset and training (Part 1)  
5. Run 02_BP_from_scratch.ipynb to BP implementation (Part 2) 
6. Run 03_model_comparison.ipynb to obtain and compare predictions using three models (Part 3)
6. Run 04_optional_par1.ipynb for BPF regulazation (Part 4 - Option 1)

```
### OPTION 2: Python scripts

4. Load and training script (Part 1):  
    ```
    python src/data_generators/NeuronalDataset.py
    ```

5. Implementation of BP (Part 2):  
    ```
    python src/neuralnet/BP_Multilayer.py
    ```

6. Model comparison (Part 3):  
    ```
    python src/neuralnet/model_comparison.py
    ```

## Features
- Dataset loading and preprocessing
- Training and evaluation scripts
- Custom BP neural network implementation
- Model comparison (BP, BP-F, MLR-F)
- Example jupyter notebooks

## Author
Victoria Joven
