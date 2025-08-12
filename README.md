# Similarity: Applying MedCLIP Architecture

## 1. Introduction
- This project aims to use [MedCLIP's](https://arxiv.org/pdf/2210.10163) architecture to primarily address the issues of data scarcity and false negatives (possibly because of unclean data)
- MedCLIP solves the problem of a lack of paired data by decoupling the data points, allowing the training set to increase to O(n^2)
- This project also aims to enhance Explainable AI (XAI) by implementing [MedCLIP-SAM](https://arxiv.org/pdf/2403.20253): Another architecture which uses saliency matching to highlight regions of interest
- **Note**: CLIP architectures consist of a vision encoder and a text encoder. The objective is to generate embeddings for each of these such that matching and similar inputs are embedded close together.

## 2. MedCLIP
- In the medical field, due to privacy, admin and financial constraints, publicly available chest scans and their corresponding medical reports (and vice versa) are rare to come by.
- Using the existing data uses a patient id as a ground truth. But if 2 patients have similar chest scans and reports, their scans will be trained to output dissimilar embeddings when they should be similar.
- MedCLIP is a pre-training framework that decouples and then recouples the data using existing architecture (in this case MetaMap) to generate a ground truth.
- It uses **Semantic Matching Loss** to fine tune existing models
- They fine tune the BioClinicalBERT text encoder and the ImageNET image encoder. 
![Semantic Matching Loss](images/sem_mat_loss.png)

### MedCLIP approach
1. Decouple data so we have a set of images and a set of text
2. Generate simple embeddings for each one using existing, lightweight architectures.
3. Dot product each embedding with each other one to generate similarity matrix
5. Use Semantic Matching Loss on similarity matrix after applying softmax both row-wise and column-wise.

## 3. MedCLIP-SAM
- The MedCLIP-SAM paper explores 3 points:
1. A new CLIP training architecture using a **Decoupled Hard Negative Noise Contrastive Estimation** (DHN-NCE) Loss Function
![Loss pt1](images/DHNNCE1.png)
![Loss pt2](images/DHNNCE2.png)
- Interestingly, MedCLIP-SAM uses a different architecture to MedCLIP
- They pre-train BiomedCLIP by Microsoft
- This is trained on a much larger dataset and has a vast general knowledge to help with Zero-Shot-Segmentation

2. Build an architecture for **Zero Shot Segmentation**: Highlight relevant pixels or region of image
    - Uses BiomedCLIP report/image retrieval with SAM (Segment Anything Model)
    - SAM performs actual segmentation
3. They explored a weakly supervised method of improving Zero Shot Segmentation

**Zero Shot Segmentation**: Model can segment even if it's never seen that specific input before
- In this case, this is achieved because Microsoft's BioMedCLIP has been massively pre-trained so has a huge existing knowledge base


## Procedure
**Start this phase in a jupyter notebook**
1. Import a "model" object which has the structure of the model stored (i.e. number of layers, neurons per layer, activation functions, etc)
2. Pre-trained weights stored in .pt or .pth file (available through a number of means eg .zip where you load them)
3. model.load_state_dict(torch.load("weights.pth"))
4. Train on dataset
5. Save weights: torch.save(model.state_dict(), "new_weights.pth")

**Now move to .py for pipeline development**
6. Load model as in step 1
7. Load weights from new_weights.pth

*This is why people use google colab with GPU capabilities*

## Final Structure (draft)

- models -> .py files with models as classes
- data/ -> Load and preprocess data
- training/ -> Python files to train data
- utils/ -> Logging, metrics, evaluation, help


- examples/ -> How to use the package, with Jupyter Notebooks showing the actual process of loading the data, training, etc
- configs/ -> Hyperparameters (JSON)
- tests/ -> Unit Tests

- README.md
- requirements.txt
- .gitignore
