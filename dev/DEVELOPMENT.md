# Vectorising DHN-NCE Loss function


# Applying gScoreCAM

- Given an image-text pair we aim to create a saliency mask on the image
- Without a ground truth, we use a self-supervised approach of generating a weak ground truth using the image encoder with 2 methods: gScoreCAM followed by SAM
- gScoreCAM according to its original research paper uses each convolutional mapping to augment the original input image
- The augmented image would then be encoded and we select the embedding with the highest similarity to its corresponding text
- We use the corresponding mapping to generate a bounding box
- This would be fed into SAM to create the weak ground truth.

- This encoder uses a vision transformer rather than a CNN, which uses attention heads rather than convolutions.
- These attention heads are much smaller in number than convolutional mappings in a CNN
- We have 2 solutions:
1. Load a separate CNN and use its convolutions
2. Use the attention maps of the vision transformer

- We will test both approaches and benchmark.


# 31/10/2025

- ViT only generates attention maps between 16x16 patches (equivalent of the tokenisation step)
- We cannot identify saliency maps from here
- Directly augmenting the bounding boxes would be very vanilla: Worth benchmarking
- Original gScoreCAM pipeline augments the original image by each convolutional mapping (hadamaard product)
- Need to find alternative method

**CLS Token**

- The CLS token is a dummy token which aggregates all other tokens with the attention mechanism.
- The final attention layer's CLS token is the input to the fully connected layer at the end
- Potentially leverage the CLS token


**Solution**

1. Obtain final layer CLS token
2. Apply mask over image patch-wise and obtain each CLS token
3. Return mask which maximises CLS token similarity

- This is a very brute-force method which iterates over several possibilities.
- There should be a way to choose from a list of masks (similar to how convolutional mappings are used in regular gScoreCAM)


**Statistics**

- Statistics in consideration are IOU, DSC and AUC (referenced in stat_test_methods.md)
- Looking at the statistics and metrics (**Table 3**) from MedCLIP-SAM paper, the segmentation accuracy of the proposed method exceeds the fully supervised ResUNet (CNN) for both breast ultrasounds and brain MRIs.
- ResUNet seems to be almost perfect for Lung X-rays, and the proposed method seems pretty far off.

- **Table 2** from MedCLIP-SAM indicates that in all instances, the proposed, fine-tuned pipeline always exceeds the pretrained BiomedCLIP model, and that gScoreCAM in every single instance exceeds the performance of GradCAM by a significant margin


- Even though the original gScoreCAM talks about how bad ViT architectures are compared to CNN architectures when using this method, when applied to a fine-tuned biomedclip model, the Vision Transformer exceeds ResUNet (CNN) in segmenting breast ultrasounds and brain MRIs.
- gScoreCAM paper evaluates each architecture on regular images
- Possibly the difference in the type of image (medical) leads to the ViT architecture performing better than CNNs
