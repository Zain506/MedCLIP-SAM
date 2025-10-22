# MedCLIP-SAM -- Paper Implementation

This project aims to implement MedCLIP-SAM: A contrastive learning model aiming to map multimodal data - images (X-rays scans) and text (radiology reports) into a shared embedding space. This works by fine-tuning encoders using a specific loss funtion. Then to enhance its explainability, we apply the SAM (Segment Anything Model) to generate a saliency map of the scan based on its corresponding report.

---

# Installation

Create and activate a virtual environment using uv

```bash
uv sync
```

If developing, use ```bash
uv sync --all-extras
```

---

# Intended Repository Structure

```
├── notebooks/ # Exploration notebooks and evaluation metrics
├── src/ # Core implementation (models, training, utilities, etc.)
├── tests/ # Unit tests for individual components
├── main.py # Entry point for running experiments
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files and directories
```
