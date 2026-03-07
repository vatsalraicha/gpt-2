# Q&A Pairs — Batch 5 (Review)

Mark each pair: **APPROVE**, **EDIT**, or **REJECT**

---

## Practical Deep Learning (201–215)

**Q201.** What is early stopping?

**A201.** Early stopping is a regularization technique that monitors the model's performance on a validation set during training and stops training when performance begins to degrade. This prevents the model from overfitting by selecting the checkpoint that achieved the best validation performance. A patience parameter determines how many epochs of no improvement to tolerate before stopping.

---

**Q202.** What is the difference between online and offline learning?

**A202.** In offline learning, also called batch learning, the model is trained on the entire dataset at once and deployed as a fixed model. In online learning, the model is updated incrementally as new data arrives, adapting to changes in the data distribution over time. Online learning is useful when data arrives in a continuous stream or when the dataset is too large to fit in memory.

---

**Q203.** What is a depthwise separable convolution?

**A203.** A depthwise separable convolution factorizes a standard convolution into two steps: a depthwise convolution that applies a single filter to each input channel independently, and a pointwise convolution that combines the outputs using 1x1 convolutions. This factorization dramatically reduces the number of parameters and computations while maintaining similar performance, making it the basis for efficient architectures designed for mobile and embedded devices.

---

**Q204.** What is global average pooling?

**A204.** Global average pooling replaces fully connected layers at the end of a convolutional neural network by computing the average of each feature map across all spatial positions. This reduces each feature map to a single value, drastically reducing the number of parameters and the risk of overfitting. Global average pooling also enforces a natural correspondence between feature maps and categories, improving interpretability.

---

**Q205.** What is batch size and how does it affect training?

**A205.** Batch size is the number of training examples processed in a single forward and backward pass. Smaller batch sizes introduce more noise into gradient estimates, which can act as regularization and help escape local minima, but they are less computationally efficient. Larger batch sizes provide more stable gradient estimates and better utilize parallel hardware, but may converge to sharper minima that generalize less well. The effective batch size can be increased through gradient accumulation.

---

**Q206.** What is weight decay?

**A206.** Weight decay is a regularization technique that adds a penalty proportional to the magnitude of the model's weights to the loss function during each update step. It is mathematically equivalent to L2 regularization when using standard gradient descent. Weight decay prevents the model from relying too heavily on any single feature by keeping the weights small, which typically leads to smoother decision boundaries and better generalization.

---

**Q207.** What is the difference between model parameters and hyperparameters?

**A207.** Model parameters are the values that the model learns during training, such as weights and biases in a neural network. Hyperparameters are set before training begins and control the learning process, such as the learning rate, batch size, number of layers, and regularization strength. While parameters are optimized automatically by the training algorithm, hyperparameters must be chosen by the practitioner or through automated search methods.

---

**Q208.** What is gradient clipping?

**A208.** Gradient clipping limits the magnitude of gradients during training to prevent the exploding gradient problem, where gradients become extremely large and cause unstable parameter updates. It works by scaling down the gradient vector when its norm exceeds a threshold. This is particularly important for training recurrent neural networks and transformers, where long sequences can lead to large gradient accumulations through backpropagation.

---

**Q209.** What is the warm-up phase in training?

**A209.** The warm-up phase gradually increases the learning rate from a small initial value to the target learning rate over the first few steps or epochs of training. This prevents large, destabilizing parameter updates early in training when the model's weights are still randomly initialized. After the warm-up phase, the learning rate typically follows a decay schedule. Warm-up is commonly used when training transformers and other large models.

---

**Q210.** What is cosine annealing?

**A210.** Cosine annealing is a learning rate schedule that decreases the learning rate following a cosine curve over the course of training. The learning rate starts high and smoothly decreases to a minimum value, which can be near zero. This schedule allows the model to make large updates early in training when it is far from a good solution and smaller, more precise updates later as it approaches convergence. Some variants restart the cosine schedule periodically.

---

**Q211.** What is mixed precision training and when should it be used?

**A211.** Mixed precision training uses both half-precision and full-precision floating point arithmetic during training to reduce memory usage and increase speed. Computations like matrix multiplications are performed in half precision, while operations that require numerical stability like loss scaling and parameter updates use full precision. This technique should be used when training on GPUs with dedicated half-precision hardware support, as it can approximately double training throughput.

---

**Q212.** What is the difference between fine-tuning and feature extraction?

**A212.** In feature extraction, the pretrained model's weights are frozen and only a new classification head is trained on the task-specific data. In fine-tuning, the entire pretrained model or selected layers are updated during training. Feature extraction is faster and requires less data since fewer parameters are being trained, while fine-tuning can achieve better performance by adapting the pretrained representations to the specific characteristics of the target task.

---

**Q213.** What is layer freezing in transfer learning?

**A213.** Layer freezing selectively prevents certain layers of a pretrained model from being updated during fine-tuning. Typically, earlier layers that capture general features are frozen while later layers that capture task-specific features are fine-tuned. Gradual unfreezing progressively unfreezes layers from top to bottom during training, allowing the model to adapt at each level while preserving the general knowledge in earlier layers.

---

**Q214.** What is the difference between token-level and sequence-level tasks?

**A214.** Token-level tasks require producing an output for each token in the input sequence, such as named entity recognition or part-of-speech tagging. Sequence-level tasks require producing a single output for the entire input sequence, such as sentiment classification or document categorization. Language modeling is a token-level task where the model predicts the next token at each position, while text classification is a sequence-level task.

---

**Q215.** What is teacher forcing in training?

**A215.** Teacher forcing is a training technique for sequence generation models where the model receives the ground truth output at each time step as input for the next prediction, rather than using its own previous predictions. This speeds up training and stabilizes learning because the model always sees correct context. However, it can create a mismatch between training and inference, where the model must use its own predictions, leading to error accumulation.

---

## Mathematical Foundations (216–228)

**Q216.** What is a vector space?

**A216.** A vector space is a mathematical structure consisting of a set of vectors that can be added together and multiplied by scalars according to specific rules. Vectors in a vector space can represent data points, features, or model parameters. Understanding vector spaces is fundamental to machine learning because operations like matrix multiplication, projection, and distance computation that are central to learning algorithms are defined in terms of vector spaces.

---

**Q217.** What is the dot product?

**A217.** The dot product of two vectors is the sum of the products of their corresponding elements, producing a scalar value. It measures the similarity between two vectors: a large positive dot product indicates the vectors point in similar directions, while a value near zero indicates they are nearly perpendicular. The dot product is the fundamental operation in computing attention weights, measuring embedding similarity, and implementing many machine learning algorithms.

---

**Q218.** What is cosine similarity?

**A218.** Cosine similarity measures the cosine of the angle between two vectors, producing a value between negative one and one. Unlike the dot product, cosine similarity is normalized by the magnitudes of the vectors, so it measures directional similarity regardless of vector length. It is widely used for comparing embeddings, computing attention scores, and measuring document similarity in information retrieval.

---

**Q219.** What is the chain rule in calculus?

**A219.** The chain rule is a formula for computing the derivative of a composite function. It states that the derivative of a composed function equals the product of the derivatives of each component function. The chain rule is the mathematical foundation of backpropagation, which computes the gradient of the loss with respect to each parameter by applying the chain rule repeatedly through the layers of the network.

---

**Q220.** What is a covariance matrix?

**A220.** A covariance matrix is a symmetric matrix where each element represents the covariance between two features in a dataset. The diagonal elements are the variances of individual features, while the off-diagonal elements measure how pairs of features vary together. The covariance matrix is used in principal component analysis, Gaussian distributions, and many other statistical methods to understand the relationships between variables.

---

**Q221.** What is singular value decomposition?

**A221.** Singular value decomposition factors any matrix into the product of three matrices: a left orthogonal matrix, a diagonal matrix of singular values, and a right orthogonal matrix. The singular values are ordered from largest to smallest and indicate the importance of each component. SVD is used in dimensionality reduction, matrix compression, recommender systems, and analyzing the rank and effective dimensionality of weight matrices in neural networks.

---

**Q222.** What is the difference between convex and non-convex optimization?

**A222.** In convex optimization, the loss function has a single global minimum, and any local minimum found by the optimizer is guaranteed to be the global minimum. In non-convex optimization, the loss function can have multiple local minima, saddle points, and plateaus. Neural network training is a non-convex optimization problem, which is why techniques like momentum, adaptive learning rates, and stochastic noise from mini-batches are important for finding good solutions.

---

**Q223.** What is the softmax function mathematically?

**A223.** The softmax function takes a vector of real-valued scores and converts it into a probability distribution where each element is positive and all elements sum to one. Each element is computed as the exponential of the input value divided by the sum of all exponentials. It is used as the final activation function in classification networks and in the attention mechanism to convert raw attention scores into normalized attention weights.

---

**Q224.** What is the information gain criterion?

**A224.** Information gain measures the reduction in entropy achieved by splitting a dataset on a particular feature. Entropy quantifies the uncertainty in a set of labels, and information gain calculates how much that uncertainty decreases after the split. Decision tree algorithms use information gain to select the best feature and threshold at each node, choosing the split that most effectively separates the data into homogeneous groups.

---

**Q225.** What is the Central Limit Theorem?

**A225.** The Central Limit Theorem states that the average of a large number of independent random variables, regardless of their individual distributions, will approximately follow a normal distribution. This result is fundamental to statistics because it justifies the use of normal distribution-based methods even when the underlying data is not normally distributed. It explains why many natural measurements follow a bell-shaped curve and forms the basis for confidence intervals and hypothesis testing.

---

**Q226.** What is the law of large numbers?

**A226.** The law of large numbers states that as the number of observations increases, the sample average converges to the expected value. This fundamental result in probability theory guarantees that with enough data, empirical estimates become arbitrarily close to the true population parameters. It justifies the use of sample statistics to estimate population quantities and underpins the reliability of machine learning models trained on large datasets.

---

**Q227.** What is a kernel function?

**A227.** A kernel function computes the inner product of two vectors in a higher-dimensional feature space without explicitly transforming the vectors into that space. This technique, known as the kernel trick, allows algorithms like support vector machines to learn non-linear decision boundaries while maintaining the computational efficiency of linear methods. Common kernel functions include the polynomial kernel and the radial basis function kernel.

---

**Q228.** What is the difference between parametric and non-parametric density estimation?

**A228.** Parametric density estimation assumes the data follows a known distribution family, like a Gaussian, and estimates the parameters of that distribution from the data. Non-parametric density estimation makes no assumptions about the functional form of the distribution and lets the data determine the shape. Kernel density estimation is a common non-parametric method that places a smooth kernel function at each data point and sums them to estimate the probability density.

---

## Security, Ethics, and Practical Considerations (229–240)

**Q229.** What is adversarial machine learning?

**A229.** Adversarial machine learning studies how machine learning models can be attacked and defended against malicious inputs. Adversarial examples are inputs that have been deliberately modified to cause the model to make incorrect predictions, often with changes that are imperceptible to humans. Understanding these vulnerabilities is essential for deploying models in security-critical applications like malware detection, autonomous driving, and authentication systems.

---

**Q230.** What is data poisoning?

**A230.** Data poisoning is an attack where an adversary manipulates the training data to compromise the learned model. By injecting carefully crafted malicious examples into the training set, an attacker can cause the model to learn incorrect patterns, such as misclassifying certain inputs or creating backdoors that can be triggered at inference time. Defending against data poisoning requires careful data validation and monitoring of the training process.

---

**Q231.** What is differential privacy?

**A231.** Differential privacy is a mathematical framework that provides formal guarantees about the privacy of individual data points in a dataset. It works by adding calibrated noise to computations or model updates, ensuring that the output does not reveal whether any specific individual's data was included in the dataset. Differential privacy is used in machine learning to train models on sensitive data while protecting the privacy of individuals.

---

**Q232.** What is algorithmic fairness?

**A232.** Algorithmic fairness addresses the concern that machine learning models may discriminate against certain groups based on sensitive attributes like race, gender, or age. Bias can enter through training data that reflects historical discrimination or through features that serve as proxies for protected attributes. Ensuring fairness requires defining appropriate fairness metrics, auditing models for disparate impact, and implementing techniques to mitigate bias.

---

**Q233.** What is model interpretability versus explainability?

**A233.** Model interpretability refers to the degree to which a human can understand the cause of a model's decisions. Simple models like decision trees and linear regression are inherently interpretable because their decision process is transparent. Model explainability refers to techniques that provide post-hoc explanations for the decisions of complex, opaque models like deep neural networks. Both are important for building trust and ensuring accountability in AI systems.

---

**Q234.** What is intrusion detection using machine learning?

**A234.** Intrusion detection systems use machine learning to identify unauthorized access or malicious activity in computer networks. They analyze network traffic patterns, system logs, and user behavior to detect deviations from normal activity. Approaches include supervised methods that learn from labeled examples of known attack types and unsupervised methods that detect anomalous behavior that may indicate previously unknown attacks.

---

**Q235.** What is user entity behavior analytics?

**A235.** User entity behavior analytics uses machine learning to model the normal behavior patterns of users and entities in a network. By establishing a baseline of typical activities, the system can detect deviations that may indicate compromised accounts, insider threats, or policy violations. This approach is particularly effective because it can identify subtle, previously unknown threats that rule-based systems would miss.

---

**Q236.** What is the role of SQL in data analytics?

**A236.** SQL is the standard language for querying and manipulating data stored in relational databases. In data analytics, SQL is used to extract, filter, aggregate, and join data from multiple tables to answer business questions. Data scientists use SQL to prepare datasets for machine learning by selecting relevant features, computing aggregations, and filtering training data. Proficiency in SQL is essential for working with the structured data that forms the foundation of many analytical pipelines.

---

**Q237.** What is the MapReduce paradigm?

**A237.** MapReduce is a programming model for processing large datasets in parallel across a distributed cluster. The map phase applies a function to each element of the data independently, producing intermediate key-value pairs. The reduce phase aggregates these intermediate results by key to produce the final output. While MapReduce has been largely superseded by more flexible frameworks like Apache Spark, its fundamental concepts of parallel data processing remain central to distributed computing.

---

**Q238.** What is model calibration?

**A238.** Model calibration measures whether a model's predicted probabilities match the actual frequencies of outcomes. A well-calibrated model that predicts a 70 percent probability for an event should be correct approximately 70 percent of the time. Neural networks are often poorly calibrated, tending to be overconfident in their predictions. Post-hoc calibration techniques like Platt scaling and temperature scaling adjust the predicted probabilities to improve calibration.

---

**Q239.** What is active learning?

**A239.** Active learning is a training strategy where the model selects which examples should be labeled next, rather than learning from a randomly sampled training set. The model identifies the examples it is most uncertain about and requests labels for those, maximizing the information gained from each labeled example. This approach significantly reduces the amount of labeled data needed to achieve good performance, which is valuable when labeling is expensive or time-consuming.

---

**Q240.** What is the difference between data lake and data warehouse?

**A240.** A data lake stores raw data in its native format at any scale, supporting structured, semi-structured, and unstructured data. A data warehouse stores processed, structured data that has been cleaned and organized for specific analytical queries. Data lakes are more flexible and cost-effective for storage but require more processing before analysis. The data lakehouse architecture combines the strengths of both approaches.

---

## Additional ML Concepts (241–250)

**Q241.** What is elastic net regularization?

**A241.** Elastic net regularization combines L1 and L2 regularization by adding both penalty terms to the loss function, weighted by a mixing parameter. This combines the feature selection property of L1 regularization with the stability of L2 regularization. Elastic net is particularly useful when there are correlated features, where L1 regularization alone might arbitrarily select one of the correlated features while ignoring the others.

---

**Q242.** What is the attention mechanism in sequence-to-sequence models?

**A242.** The attention mechanism in sequence-to-sequence models allows the decoder to selectively focus on different parts of the input sequence when generating each output token. At each decoding step, the mechanism computes alignment scores between the current decoder state and all encoder hidden states, then uses these scores to create a weighted combination of the encoder states. This addresses the bottleneck of compressing the entire input into a single fixed-length vector.

---

**Q243.** What is mini-batch training?

**A243.** Mini-batch training processes a small subset of the training data at each update step, rather than the entire dataset or a single example. This balances the stability of full-batch gradient computation with the regularizing noise of single-example updates. Mini-batch sizes typically range from 16 to 512 examples, chosen based on hardware memory constraints and the desired trade-off between training speed and gradient estimation quality.

---

**Q244.** What is the epoch in machine learning training?

**A244.** An epoch is one complete pass through the entire training dataset during model training. In each epoch, every training example is seen exactly once by the model. Training typically requires multiple epochs for the model to converge, with the loss generally decreasing over successive epochs. The number of epochs is a hyperparameter, and training for too many epochs can lead to overfitting while too few epochs results in an undertrained model.

---

**Q245.** What is the perceptron learning algorithm?

**A245.** The perceptron learning algorithm is one of the earliest machine learning algorithms for binary classification. It iteratively adjusts weights by comparing predictions to true labels: when the prediction is incorrect, the weights are updated in the direction that would correct the error. The perceptron converges to a perfect classifier if the data is linearly separable, but cannot solve problems that require non-linear decision boundaries, which led to the development of multilayer networks.

---

**Q246.** What is non-negative matrix factorization?

**A246.** Non-negative matrix factorization decomposes a matrix into two matrices with the constraint that all elements must be non-negative. This constraint produces parts-based representations where each component captures an additive part of the original data. It is particularly useful for topics modeling in text, where topics are represented as non-negative combinations of words, and for image analysis, where parts correspond to meaningful visual components.

---

**Q247.** What is the difference between generative and discriminative models?

**A247.** Discriminative models learn the boundary between classes directly by modeling the conditional probability of the label given the input. Examples include logistic regression, support vector machines, and neural network classifiers. Generative models learn the joint probability distribution of inputs and labels, modeling how the data was generated. Examples include naive Bayes and hidden Markov models. Generative models can generate new data samples, while discriminative models typically achieve higher classification accuracy.

---

**Q248.** What is normalization in the context of data preprocessing?

**A248.** Normalization scales numerical features to a standard range, typically zero to one or negative one to one. This is important because features with larger scales can dominate the learning process in algorithms that are sensitive to feature magnitudes. Min-max normalization subtracts the minimum and divides by the range, while z-score standardization subtracts the mean and divides by the standard deviation to produce features with zero mean and unit variance.

---

**Q249.** What is the receptive field in a convolutional neural network?

**A249.** The receptive field of a neuron in a convolutional neural network is the region of the input that affects the neuron's output. Each convolutional and pooling layer increases the receptive field of neurons in subsequent layers. Deeper layers have larger receptive fields, allowing them to capture broader spatial patterns. The design of the network architecture, including kernel sizes, strides, and the number of layers, determines how the receptive field grows.

---

**Q250.** What is the difference between model complexity and model capacity?

**A250.** Model complexity generally refers to the number of parameters or the flexibility of a model to fit different functions, while model capacity refers to the range of functions the model can represent. A model with higher capacity can fit more complex patterns but is also more prone to overfitting. Techniques like regularization, dropout, and early stopping are used to control the effective capacity of a model to balance the ability to fit the training data with generalization to new data.

---

*End of Batch 5 — Q201–Q250 for review*

---

**All 250 Q&A pairs complete. Review all 5 batch files and let me know your approvals/edits/rejections.**
