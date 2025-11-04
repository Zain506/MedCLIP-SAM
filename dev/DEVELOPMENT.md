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


# 4 Nov

- We cannot apply the regular gScoreCAM method of augmenting images by each convolution
- Computationally inefficient to brute force every possible augmentation to generate a saliency map
- NAIVE APPROACH: Augment each image by filtering out all but 1 patch. Patchwise augmentation to find most relevant patch
- If attention scores were available, could apply filtering with each attention score
- gScoreCAM paper highlights some alternatives for ViT architectures:
    1. White Box: Using gradients or activation maps
    2. Black Box: Perturbing the input

**gScoreCAM S3.5: CLIP Networks**

- Reshape the embedding of the target layer
- Choose a valid channel closest to the CLIP prediction layer
- For ViT-B/32: Second last ResidualAttentionBlock in VisionTransformer
- Output dimension 50x1x768.
- Exclude [CLS] token and then reshape into 7x7x768 for CAM-based visualisations.
- 2nd to last layer is used because last layer's gradients are 0 except CLS token. And CLS token is used for prediction


**Algorithm**:

1. Apply forward pass and store patch embeddings from 2nd last ResidualAttentionBlock and compute loss. Ignore CLS token.
2. Apply backward pass and compute gradients of similarity wrt each latent dimension in each patch
3. We have 7x7x768 gradients: Gradient per latent feature in each patch
4. Obtain the top k gradients - and top k latent features by taking average gradient across patches.
5. Each of the k latent features is a channel (according to section 3.5)
6. From the original 7x7x768 activation, turn into 7x7xk by filtering only top k channels/dimensions.
7. Upsample each 7x7 map to get a matrix the same size as the image per channel (HxWxk)
8. Apply a Hadamaard product to augment the input image by each upsampled map corresponding to a channel.
9. Compute the similarity between each augmented image and the text embedding: CLIP Score. Convert into weighting (eg probability distribution)
10. Apply a linear combination between the clip scores and the masks onto the image to get a final heatmap


