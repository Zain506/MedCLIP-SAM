# MedCLIP-SAM -- Paper Implementation

- This repository is a personal project aiming to explore the [MedCLIP-SAM](https://arxiv.org/pdf/2403.20253) research paper.
- The goal is to understand the algorithms and get hands-on experience with libraries such as PyTorch.
- I will aim to potentially improve on the original approach if weaknesses are detected

---

## Project Description

This project aims to:
- Re-implement the techniques and models described in the target research paper (including the model fine-tuning)
- Learn by building -- focussing on how each component works under the hood
- Improve my ability to interpret research papers and apply them
- Explore modifications or improvements
- Document findings

> This is a learning-oriented project

---

## Tech Stack

- Python
- PyTorch - model training and tensor operations
- HuggingFace Transformers - loading datasets and pretrained models
- Openclip - Import pretrained models

---

## Notebooks Directory

- The majority of the work is currently in the notebooks directory
- Here I am implementing both stages of the research paper:
	1. Fine-tuning pre-existing BiomedCLIP
	2. Saliency Map generation on images
- The first notebook generates the pretrained weights that the second notebook imports
- Each notebook can be run on Google Drive, and has code blocks aimed at exporting and importing to Google Drive
- If run locally, you will need to download the weights and then run the block which imports them under the same file name

---

##  How to run

1. Clone the Repository
```bash
git clone https://github.com/Zain506/MedCLIP-SAM.git
cd MedCLIP-SAM/
```

2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the code in notebooks

```bash
cd notebooks
jupyter lab
```

---

## Intended Repository Structure

```
├── notebooks/ # Exploration notebooks and evaluation metrics
├── src/ # Core implementation (models, training, utilities, etc.)
├── tests/ # Unit tests for individual components
├── main.py # Entry point for running experiments
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files and directories
```
