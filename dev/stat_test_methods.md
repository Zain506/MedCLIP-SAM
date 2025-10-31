# Methods of obtaining metrics



1. Paired McNemar Statistical Tests

- Referenced in the MedCLIP-SAM paper
- Compare performance of 2 classifiers on teh same dataset
- Mainly for categorical predictions
- Non-parametric test designed for paired nominal data
- From confusion matrix, compute McNemar Test Statistic (approximately follows a Chi-squared distr.)
$$H_0: \text{Both models perform equally}$$
$$H_1: \text{Models perform differently}$$

- use 5% significance


2. IOU (intersection over union)

- Measures overlap of 2 sets

3. DSC (Dice Similarity Coefficient)

- F1 score for segmentation
- Similar to IOU

4. AUC: Area under ROC curve
