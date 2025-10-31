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
