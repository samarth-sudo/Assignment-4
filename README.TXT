Assignment 4
Group:- 
(1)Samarth Singh(samarths@bu.edu)
(2)Adwait Kulkarni (adk1361@bu.edu)

#Questions and Answers

Solution 1. Difference between classification and segmentation, and potential conflicts:

Classification and segmentation are related but distinct tasks in computer vision and image processing.

- Classification** involves assigning a single label or category to an entire image or object. For example, an image could be classified as "car" or "pedestrian".
- Segmentation**, on the other hand, involves partitioning an image into multiple segments or regions, and assigning a label to each pixel or group of pixels. The goal is to identify the boundaries and shapes of different objects or regions within the image.

Potential conflicts can arise because classification treats the entire image as a single entity, while segmentation requires pixel-level or region-level labeling. Segmentation is a more fine-grained task that requires localization and delineation of object boundaries, which is not necessary for classification.

Solution 2. Fully Convolutional Networks (FCNs) and their versions:

FCNs address the conflict between classification and segmentation by adapting convolutional neural networks (CNNs), which were initially designed for classification, to perform pixel-wise prediction and segmentation.

- The original FCN paper (Long et al., 2015) introduced a fully convolutional architecture that replaced the fully connected layers in traditional CNNs with convolutional layers, allowing for dense predictions and preserving spatial information.
- Different versions of FCNs have been proposed to balance the trade-off between spatial resolution and context:
  - FCN-32s: Upsamples the final feature maps by a factor of 32 to produce segmentation maps.
  - FCN-16s: Combines predictions from the final feature maps and earlier feature maps for better localization.
  - FCN-8s: Combines predictions from the final, pool4, and pool3 layers for even better localization.

These variants strike different balances between preserving spatial resolution (better localization) and incorporating broader context (better recognition).

Solution 3. Evaluation metrics: pixel accuracy, IU, mean IU, and frequency-weighted IU:

- Pixel accuracy: The proportion of correctly classified pixels over the total number of pixels. However, this metric can be misleading when the distribution of classes is imbalanced.
- Intersection over Union (IU): Also known as the Jaccard index, it measures the overlap between the predicted and ground truth segmentation for a single class. IU = True Positives / (True Positives + False Positives + False Negatives).
- Mean IU: The average of IU values across all classes, treating each class equally.
- Frequency-weighted IU: Calculates the IU for each class and then weights the IU values by the frequency of each class in the dataset. This gives more weight to common classes and less weight to rare classes.

Mean IU is a more balanced metric, while frequency-weighted IU accounts for class imbalance in the dataset.

Solution 4. Limitations of FCNs and potential improvements:

While FCNs have been successful in semantic segmentation, there are several limitations and areas for potential improvement:

- Resolution and localization: FCNs still struggle with precise localization of object boundaries, especially for small objects or thin structures.
- Context and global information: Although FCNs incorporate some context, they may still struggle with incorporating broader scene context and global information.
- Real-time performance: FCNs can be computationally expensive, limiting their applicability in real-time scenarios.

Potential directions for improvement include:

- Multi-scale and multi-level feature fusion: Combining features from different scales and levels of the network to capture both local and global information.
- Attention mechanisms: Incorporating attention mechanisms to focus on relevant regions and suppress irrelevant information.
- Instance segmentation: Extending FCNs to perform instance-level segmentation, separating individual object instances within the same class.
- Efficient architectures: Exploring more efficient and lightweight architectures for real-time applications, such as mobile devices or embedded systems.
- Weakly-supervised or unsupervised learning: Developing methods that can learn from weaker forms of supervision or unsupervised data to reduce the need for extensive manual annotations.
