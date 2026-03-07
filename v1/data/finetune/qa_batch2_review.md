# Q&A Pairs — Batch 2 (Review)

Mark each pair: **APPROVE**, **EDIT**, or **REJECT**

---

## PyTorch and Implementation (51–62)

**Q51.** What is a tensor in PyTorch?

**A51.** A tensor is the fundamental data structure in PyTorch, similar to a multidimensional array. Tensors can represent scalars, vectors, matrices, and higher-dimensional data. They support automatic differentiation, which is essential for training neural networks through backpropagation. Tensors can be moved between CPU and GPU for accelerated computation.

---

**Q52.** How do you represent real-world data as tensors?

**A52.** Real-world data must be converted into tensors before it can be processed by a neural network. Images are represented as three-dimensional tensors with dimensions for channels, height, and width. Text is converted into sequences of integer token indices and then mapped to embedding vectors. Tabular data is represented as a two-dimensional tensor where rows are samples and columns are features.

---

**Q53.** What is automatic differentiation in PyTorch?

**A53.** Automatic differentiation is PyTorch's mechanism for computing gradients of the loss function with respect to all model parameters. When operations are performed on tensors with gradient tracking enabled, PyTorch builds a computational graph that records each operation. During backpropagation, this graph is traversed in reverse to compute the gradient of each parameter using the chain rule.

---

**Q54.** What is a PyTorch DataLoader?

**A54.** A DataLoader is a PyTorch utility that provides an efficient way to iterate over a dataset in batches. It handles batching, shuffling, and parallel data loading using multiple worker processes. The DataLoader takes a Dataset object that defines how to access individual samples and returns batches of tensors ready for the model to process.

---

**Q55.** How are pretrained networks used in PyTorch?

**A55.** PyTorch provides access to pretrained networks that have been trained on large datasets like ImageNet. These pretrained models can be loaded and used directly for inference or adapted to new tasks through fine-tuning. The typical workflow involves loading the pretrained model, replacing the final classification layer to match the number of target classes, and then training on the new dataset with a lower learning rate.

---

**Q56.** What is the difference between model.train() and model.eval() in PyTorch?

**A56.** In PyTorch, calling model.train() sets the model to training mode, where layers like dropout and batch normalization behave differently than during inference. Dropout randomly zeroes neurons during training but is disabled during evaluation. Calling model.eval() switches the model to evaluation mode, ensuring that predictions are deterministic and that batch normalization uses running statistics rather than batch statistics.

---

**Q57.** What is the optimizer's role in PyTorch training?

**A57.** The optimizer in PyTorch is responsible for updating model parameters based on the computed gradients. After computing the loss and calling loss.backward() to compute gradients, the optimizer's step() method updates each parameter according to the optimization algorithm. Common optimizers include SGD, Adam, and Adadelta, each with different strategies for adjusting the learning rate and momentum.

---

**Q58.** What is gradient accumulation?

**A58.** Gradient accumulation is a technique that allows training with a larger effective batch size than what fits in memory. Instead of updating parameters after every batch, gradients are accumulated over multiple forward and backward passes before performing a single parameter update. This is particularly useful when training large models on hardware with limited memory, as it simulates the effect of a larger batch size.

---

**Q59.** What is mixed precision training?

**A59.** Mixed precision training uses both 16-bit and 32-bit floating point numbers during training to reduce memory usage and increase computational speed. The forward pass and gradient computation use 16-bit precision, while the parameter updates and loss scaling use 32-bit precision to maintain numerical stability. This technique can significantly speed up training on modern GPUs with dedicated hardware support for 16-bit operations.

---

**Q60.** What is a learning rate scheduler?

**A60.** A learning rate scheduler automatically adjusts the learning rate during training according to a predefined schedule. Common strategies include step decay, which reduces the learning rate by a factor at fixed intervals, and cosine annealing, which smoothly decreases the learning rate following a cosine curve. Using a scheduler often leads to better convergence than a fixed learning rate because the model can take larger steps early in training and smaller, more precise steps later.

---

**Q61.** What is weight initialization in neural networks?

**A61.** Weight initialization determines the starting values of model parameters before training begins. Poor initialization can lead to vanishing or exploding gradients, making training difficult or impossible. Common strategies include Xavier initialization, which scales weights based on the number of input and output units, and Kaiming initialization, which is designed specifically for layers with ReLU activation functions.

---

**Q62.** What is the purpose of validation during training?

**A62.** Validation is the process of evaluating a model on a held-out portion of the data during training to monitor its generalization performance. By comparing training loss to validation loss, we can detect overfitting early: if training loss continues to decrease while validation loss starts to increase, the model is memorizing rather than learning. Early stopping uses validation performance to determine when to stop training.

---

## Data Science and Preprocessing (63–74)

**Q63.** What is data preprocessing in machine learning?

**A63.** Data preprocessing involves transforming raw data into a clean and suitable format for machine learning algorithms. Common steps include handling missing values, encoding categorical variables, scaling numerical features, and splitting the data into training and test sets. Building good training datasets through careful preprocessing is essential because the quality of the input data directly affects model performance.

---

**Q64.** What is feature scaling and why is it important?

**A64.** Feature scaling transforms the values of different features to a similar range, which is important for many machine learning algorithms. Standardization transforms features to have zero mean and unit variance, while min-max scaling transforms features to a fixed range like zero to one. Algorithms that rely on distance calculations, such as k-nearest neighbors and support vector machines, are particularly sensitive to the scale of features.

---

**Q65.** How do you handle missing data?

**A65.** Missing data can be handled through several approaches depending on the nature and extent of the missing values. Deletion removes rows or columns with missing values, which is appropriate when the proportion of missing data is small. Imputation fills in missing values using strategies like mean, median, or mode of the available values. More sophisticated methods use models to predict missing values based on the observed data.

---

**Q66.** What is one-hot encoding?

**A66.** One-hot encoding is a method for transforming nominal categorical features into a numerical representation that machine learning algorithms can process. Each unique category is represented as a binary vector with a one in the position corresponding to that category and zeros elsewhere. While effective, one-hot encoding can create very high-dimensional representations when the number of unique categories is large.

---

**Q67.** What is the difference between training, validation, and test sets?

**A67.** The training set is used to fit the model parameters, the validation set is used to tune hyperparameters and monitor for overfitting during training, and the test set is used for final evaluation of the model's performance on unseen data. It is critical that the test set is never used during model development, as any exposure would compromise the estimate of how the model will perform on truly new data.

---

**Q68.** What is data augmentation?

**A68.** Data augmentation is a technique for increasing the effective size and diversity of a training dataset by applying transformations to existing samples. In computer vision, common augmentations include random rotations, flips, crops, and changes in brightness or contrast. Data augmentation acts as a form of regularization by exposing the model to more variations, which helps prevent overfitting and improves generalization.

---

**Q69.** What is exploratory data analysis?

**A69.** Exploratory data analysis is the process of examining and visualizing data to understand its structure, identify patterns, detect anomalies, and formulate hypotheses before applying machine learning algorithms. It involves computing summary statistics, creating visualizations like histograms and scatter plots, and examining relationships between variables. This step is essential for understanding the data and making informed decisions about preprocessing and modeling.

---

**Q70.** What are evaluation metrics for classification?

**A70.** Evaluation metrics for classification measure how well a model distinguishes between classes. Accuracy measures the proportion of correct predictions, but can be misleading with imbalanced classes. Precision measures the proportion of positive predictions that are correct, while recall measures the proportion of actual positives that are correctly identified. The F1 score is the harmonic mean of precision and recall, providing a balanced measure of both.

---

**Q71.** What is the confusion matrix?

**A71.** A confusion matrix is a table that summarizes the performance of a classification model by showing the counts of true positives, true negatives, false positives, and false negatives. It provides a complete picture of how the model's predictions compare to the actual labels. From the confusion matrix, we can compute metrics like accuracy, precision, recall, and the F1 score.

---

**Q72.** What is the ROC curve?

**A72.** The ROC curve, or Receiver Operating Characteristic curve, is a visualization that shows the trade-off between the true positive rate and the false positive rate at different classification thresholds. The area under the ROC curve, called the AUC, provides a single number summarizing the model's ability to distinguish between classes across all thresholds. An AUC of 1.0 indicates perfect classification, while 0.5 indicates performance no better than random.

---

**Q73.** What is the difference between parametric and nonparametric models?

**A73.** Parametric models assume a specific functional form for the relationship between inputs and outputs, characterized by a fixed number of parameters. Examples include linear regression and logistic regression. Nonparametric models make fewer assumptions about the data distribution and can have a flexible number of parameters that grows with the data. Examples include k-nearest neighbors and decision trees.

---

**Q74.** What is the curse of dimensionality?

**A74.** The curse of dimensionality refers to the problems that arise when working with high-dimensional data. As the number of features increases, the volume of the feature space grows exponentially, making the available data increasingly sparse. This sparsity means that more data is needed to achieve reliable statistical estimates, distances between points become less meaningful, and many algorithms become computationally intractable.

---

## Ensemble Methods and Advanced ML (75–86)

**Q75.** What is ensemble learning?

**A75.** Ensemble learning combines multiple models to produce predictions that are typically more accurate and robust than any single model. The key insight is that different models may make different errors, and combining their predictions can cancel out individual mistakes. Common ensemble strategies include bagging, which trains models on different random subsets of the data, and boosting, which trains models sequentially to correct the errors of previous models.

---

**Q76.** What is XGBoost?

**A76.** XGBoost is a gradient boosting algorithm that has become one of the most widely used machine learning algorithms for tabular data. It builds an ensemble of decision trees sequentially, where each new tree is trained to correct the errors of the previous trees. XGBoost includes built-in regularization, handles missing values, and supports parallel computation, making it both powerful and efficient for regression and classification tasks.

---

**Q77.** What is k-nearest neighbors?

**A77.** K-nearest neighbors is a lazy learning algorithm that classifies new data points based on the majority class among the k closest training examples in the feature space. It requires no training phase; instead, all computation happens at prediction time. The choice of k affects the model's behavior: a small k makes the model sensitive to noise, while a large k makes the decision boundary smoother but may miss local patterns.

---

**Q78.** What is naive Bayes classification?

**A78.** Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the assumption that features are conditionally independent given the class label. Despite this strong assumption rarely holding in practice, naive Bayes often performs surprisingly well. It is computationally efficient, works well with high-dimensional data, and requires relatively little training data, making it popular for text classification tasks.

---

**Q79.** What is gradient boosting?

**A79.** Gradient boosting is an ensemble technique that builds models sequentially, with each new model trained to predict the residual errors of the previous ensemble. The algorithm uses gradient descent in function space, fitting new models to the negative gradient of the loss function. This approach produces strong predictive models but can be prone to overfitting if the number of trees is too large or the learning rate is too high.

---

**Q80.** What is the difference between bagging and boosting?

**A80.** Bagging and boosting are both ensemble methods that combine multiple models, but they differ in how the models are trained. Bagging trains each model independently on random bootstrap samples of the data, then averages their predictions, which reduces variance. Boosting trains models sequentially, where each model focuses on correcting the errors of previous models, which reduces bias. Random forests use bagging, while XGBoost uses boosting.

---

**Q81.** What is hyperparameter tuning?

**A81.** Hyperparameter tuning is the process of finding the optimal values for parameters that are not learned during training but set before training begins. Examples include the learning rate, the number of layers in a neural network, and the regularization strength. Common approaches include grid search, which tries all combinations from a predefined set, and random search, which samples combinations randomly from specified distributions.

---

**Q82.** What is clustering?

**A82.** Clustering is an unsupervised learning technique that groups similar data points together without using predefined labels. K-means clustering partitions data into k groups by iteratively assigning points to the nearest centroid and updating centroids. Hierarchical clustering builds a tree of nested clusters. The quality of clustering depends on the choice of algorithm, distance metric, and the number of clusters.

---

**Q83.** What is dimensionality reduction?

**A83.** Dimensionality reduction is the process of reducing the number of features in a dataset while preserving as much information as possible. Principal component analysis finds the directions of maximum variance and projects the data onto a lower-dimensional space. Other methods include t-SNE, which preserves local neighborhood structure for visualization, and autoencoders, which learn nonlinear compressed representations through neural networks.

---

**Q84.** What is anomaly detection?

**A84.** Anomaly detection identifies data points that deviate significantly from the normal behavior in a dataset. Methods range from statistical approaches that define boundaries based on the distribution of normal data to machine learning approaches that learn a model of normality and flag deviations. Applications include fraud detection, intrusion detection in cybersecurity, and quality control in manufacturing.

---

**Q85.** What is causal inference?

**A85.** Causal inference is the process of determining whether a relationship between variables is causal rather than merely correlational. It addresses questions about what would happen if we intervened on a variable, rather than just observing it. Key concepts include confounding variables, counterfactuals, and the distinction between observational and experimental studies. Methods like propensity score matching and instrumental variables help estimate causal effects from observational data.

---

**Q86.** What is Simpson's paradox?

**A86.** Simpson's paradox occurs when a trend that appears in several groups of data reverses or disappears when the groups are combined. This paradox arises because of confounding variables that influence both the treatment and the outcome. It demonstrates why careful statistical analysis and understanding of causal relationships are essential, as naive aggregation of data can lead to misleading conclusions.

---

## Computer Vision (87–94)

**Q87.** What is image classification?

**A87.** Image classification is the task of assigning a label to an image from a predefined set of categories. Convolutional neural networks have become the standard approach, using layers of learned filters to automatically extract features from images. Modern approaches use pretrained models that have been trained on millions of images and fine-tune them for specific classification tasks, achieving high accuracy even with limited training data.

---

**Q88.** What is object detection?

**A88.** Object detection goes beyond image classification by identifying the locations of objects within an image, typically by drawing bounding boxes around them. The task requires the model to both classify what objects are present and determine where they are. Architectures like YOLO process the entire image in a single pass, while region-based methods like Faster R-CNN first propose candidate regions and then classify each one.

---

**Q89.** What is image segmentation?

**A89.** Image segmentation assigns a class label to each pixel in an image, providing a detailed understanding of the scene. Semantic segmentation labels every pixel with a class, while instance segmentation additionally distinguishes between different objects of the same class. The U-Net architecture, with its encoder-decoder structure and skip connections, has been particularly successful for segmentation tasks in medical imaging.

---

**Q90.** What is a max pooling layer?

**A90.** Max pooling is an operation that reduces the spatial dimensions of feature maps by selecting the maximum value from each local region. A typical max pooling layer uses a 2x2 window with a stride of 2, reducing the height and width by half. This operation provides a form of translation invariance and reduces the number of parameters in subsequent layers, helping to prevent overfitting.

---

**Q91.** What is a residual connection?

**A91.** A residual connection, or skip connection, adds the input of a layer directly to its output, allowing the network to learn residual functions rather than complete transformations. This addresses the degradation problem where very deep networks become harder to train. Residual networks can be trained with hundreds or thousands of layers because the gradients can flow directly through the skip connections during backpropagation.

---

**Q92.** What is the difference between a stride and padding in convolutions?

**A92.** Stride determines how far the convolutional filter moves across the input at each step. A stride of one moves the filter one pixel at a time, while a stride of two skips every other position, reducing the output size. Padding adds zeros around the border of the input to control the output dimensions. Same padding preserves the spatial dimensions, while valid padding applies the filter only where it fully overlaps the input.

---

**Q93.** What is data augmentation for images?

**A93.** Data augmentation for images applies random transformations to training images to increase the diversity of the training set. Common transformations include random horizontal flips, rotations, cropping, color jittering, and scaling. These augmentations teach the model to be invariant to such transformations, improving generalization. More advanced techniques include mixup, which blends pairs of images, and cutout, which randomly masks portions of the image.

---

**Q94.** What are precision and recall in the context of detection?

**A94.** In detection tasks like medical imaging or object detection, precision measures the proportion of detected items that are truly positive, while recall measures the proportion of actual positive items that are detected. There is often a trade-off between the two: increasing sensitivity catches more true positives but may also increase false alarms. The F1 score provides a balanced measure that considers both precision and recall equally.

---

## Reinforcement Learning and Genetic Algorithms (95–100)

**Q95.** What is reinforcement learning?

**A95.** Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving rewards or penalties. The agent's goal is to learn a policy that maximizes the cumulative reward over time. Unlike supervised learning, the agent is not told the correct action but must discover which actions yield the best outcomes through trial and error.

---

**Q96.** What is the exploration-exploitation trade-off?

**A96.** The exploration-exploitation trade-off is a fundamental challenge in reinforcement learning. Exploitation means choosing the action that the agent currently believes will yield the highest reward, while exploration means trying new actions to discover potentially better strategies. Too much exploitation can lead to suboptimal policies if the agent settles on a locally good strategy, while too much exploration wastes time on poor actions.

---

**Q97.** What is deep reinforcement learning?

**A97.** Deep reinforcement learning combines reinforcement learning with deep neural networks to handle environments with large or continuous state spaces. The neural network approximates the value function or policy, allowing the agent to generalize across similar states rather than memorizing values for each individual state. Techniques like Deep Q-Networks and policy gradient methods have achieved remarkable results in game playing and robotics.

---

**Q98.** What are genetic algorithms?

**A98.** Genetic algorithms are optimization methods inspired by Darwinian evolution. They maintain a population of candidate solutions that evolve over generations through selection, crossover, and mutation. Solutions with higher fitness are more likely to be selected for reproduction, and crossover combines elements from two parent solutions to create offspring. This process iteratively improves the population toward better solutions.

---

**Q99.** What is the cross-entropy method for optimization?

**A99.** The cross-entropy method is an optimization technique that iteratively samples a population of candidate solutions from a probability distribution, evaluates their fitness, and updates the distribution to focus on the best-performing solutions. It is used in reinforcement learning as a simple but effective method for learning policies. The algorithm maintains a distribution over possible actions and gradually narrows it toward optimal behavior.

---

**Q100.** What is the difference between model-based and model-free reinforcement learning?

**A100.** In model-based reinforcement learning, the agent builds an internal model of the environment's dynamics and uses it to plan actions. In model-free reinforcement learning, the agent learns directly from experience without building an explicit model of how the environment works. Model-free methods like Q-learning are simpler and more widely used, but model-based methods can be more sample-efficient because they can simulate experiences using the learned model.

---

*End of Batch 2 — Q51–Q100 for review*
