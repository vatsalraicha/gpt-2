# Q&A Pairs — Batch 1 (Review)

Mark each pair: **APPROVE**, **EDIT**, or **REJECT**

---

## Core ML Concepts (1–12)

**Q1.** What is gradient descent?

**A1.** Gradient descent is a general technique for minimizing risk functions. It works by iteratively adjusting the parameters of a model in the direction that reduces the loss. At each step, the gradient of the loss function with respect to the parameters is computed, and the parameters are updated by subtracting a fraction of the gradient, controlled by the learning rate.

---

**Q2.** What is overfitting and why does it matter?

**A2.** Overfitting occurs when a model fits the training data too closely, capturing noise rather than the underlying pattern. An overfitted model will not make accurate predictions on new, unseen data. This is a fundamental concern in machine learning because the goal is to build models that generalize well beyond the training set.

---

**Q3.** What is the bias-variance trade-off?

**A3.** The bias-variance trade-off describes how model complexity affects predictive accuracy. A model with high bias makes strong assumptions and may underfit the data, while a model with high variance fits the training data closely but may overfit. The goal is to find a balance that minimizes total prediction error. The phenomenon of double descent has shown that this trade-off can behave unexpectedly with very complex models.

---

**Q4.** What is regularization and why is it used?

**A4.** Regularization is a technique used to prevent overfitting by adding a penalty term to the loss function that discourages overly complex models. Common forms include L2 regularization, which penalizes large parameter values, and L1 regularization, which encourages sparsity. Dropout and batch normalization are also used as forms of regularization in deep learning.

---

**Q5.** What is a loss function?

**A5.** A loss function measures how well a model's predictions match the actual target values. The concept of minimizing an empirical risk function to estimate the parameters of a model is central to machine learning. Common loss functions include mean squared error for regression and cross-entropy for classification. The choice of loss function directly affects how the model learns.

---

**Q6.** What is the difference between supervised and unsupervised learning?

**A6.** In supervised learning, the model is trained on labeled data where both the input features and the target output are provided. The goal is to learn a mapping from inputs to outputs for tasks like classification and regression. In unsupervised learning, there are no target labels. Instead, the model discovers patterns and structure in the data, such as clusters or principal components.

---

**Q7.** What is feature engineering?

**A7.** Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the model. This includes creating new features, selecting relevant ones, and transforming existing features to improve model performance. Good feature engineering often has a larger impact on results than the choice of algorithm.

---

**Q8.** What is cross-validation?

**A8.** Cross-validation is a technique for evaluating how well a model generalizes to new data. In k-fold cross-validation, the dataset is split into k subsets. The model is trained on k-1 folds and evaluated on the remaining fold, and this process is repeated k times. The average performance across all folds gives a more reliable estimate than a single train-test split.

---

**Q9.** What is the difference between classification and regression?

**A9.** Classification and regression are both supervised learning tasks. In classification, the model predicts a discrete class label, such as whether an email is spam or not. In regression, the model predicts a continuous numerical value, such as a house price. Different loss functions and evaluation metrics are used for each type of task.

---

**Q10.** What is a random variable?

**A10.** A random variable is a mathematical function that maps outcomes of a random process to numerical values. Every average value calculated from a dataset, every loss function value calculated when training a machine learning algorithm, and every parameter value of a neural network trained using a dataset are all affected by the random variation within data. Understanding random variables and probability distributions is essential for data science.

---

**Q11.** What is maximum likelihood estimation?

**A11.** Maximum likelihood estimation is a method for estimating the parameters of a probabilistic model. It works by finding the parameter values that maximize the likelihood of observing the given data. This approach is fundamental to building predictive models that explicitly account for the random component within data, and it forms the basis for many statistical and machine learning methods.

---

**Q12.** What is Bayes' theorem?

**A12.** Bayes' theorem provides a way to update the probability of a hypothesis given new evidence. It connects the prior probability of a hypothesis with the likelihood of the observed data to produce a posterior probability. Bayesian inference uses this theorem to estimate model parameters by combining prior beliefs with observed data, and techniques like Markov Chain Monte Carlo can be used to sample from the posterior distribution.

---

## Deep Learning (13–26)

**Q13.** What is backpropagation?

**A13.** Backpropagation is the process by which neural networks learn. The output layer of the network yields a loss, calculated using a loss function that compares the network's predictions to the actual targets. This loss is then propagated backward through the network using the chain rule of differentiation, computing gradients for each parameter so they can be updated to reduce the loss.

---

**Q14.** What is an activation function and why is it important?

**A14.** Activation functions are crucial to neural networks as they add the non-linearity without which, no matter how many layers a network has, it would behave like a single linear transformation. Common activation functions include ReLU, which outputs zero for negative inputs and the input value for positive inputs, and sigmoid, which squeezes values between zero and one.

---

**Q15.** What is dropout?

**A15.** Dropout is a regularization technique for neural networks where randomly selected neurons are ignored during training. At each training step, each neuron has a probability of being temporarily removed along with all its connections. This prevents neurons from co-adapting too much and forces the network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.

---

**Q16.** What is batch normalization?

**A16.** Batch normalization is a technique that normalizes the inputs to each layer of a neural network by adjusting and scaling the activations. It helps address the vanishing gradient problem and allows the use of higher learning rates, which can speed up training. Batch normalization is commonly applied alongside dropout and L2 regularization to improve training stability.

---

**Q17.** What is a convolutional neural network?

**A17.** A convolutional neural network is a type of deep learning architecture primarily used for image classification and computer vision tasks. It uses convolutional layers that apply learned filters to detect features in input images, followed by pooling layers that reduce spatial dimensions. A typical architecture contains convolutional, linear, max-pooling, and dropout layers, with activation functions like ReLU applied between layers.

---

**Q18.** What is a recurrent neural network?

**A18.** A recurrent neural network is a type of neural network designed for sequential data, where the output from a previous step is fed as input to the current step. Deep recurrent neural networks and modular structures such as long short-term memory units have made it possible to construct models that incorporate temporal dependencies directly. They have been widely used for tasks such as time series analysis, language modeling, and sequence prediction.

---

**Q19.** What is the vanishing gradient problem?

**A19.** The vanishing gradient problem occurs during backpropagation in deep neural networks when gradients become extremely small as they are propagated through many layers. This happens because the chain rule of differentiation multiplies many small values together, making it difficult for early layers to learn. Solutions include using activation functions like ReLU, applying batch normalization, and using architectures designed to mitigate this issue like LSTMs.

---

**Q20.** What is transfer learning?

**A20.** Transfer learning is the practice of using a model that has been pretrained on a large dataset and adapting it to a new, often smaller, task. Instead of training a model from scratch, the pretrained model's learned representations are fine-tuned on the target task. This approach is especially powerful when the target dataset is too small to train a deep model from scratch, and it has become the standard approach in computer vision and natural language processing.

---

**Q21.** What is the difference between a perceptron and a multilayer neural network?

**A21.** A perceptron is the simplest form of a neural network, consisting of a single layer that computes a weighted sum of its inputs and applies a threshold function to produce a binary output. A multilayer neural network, or multilayer perceptron, stacks multiple layers of neurons with activation functions between them. The additional layers allow the network to learn complex, non-linear relationships in the data that a single perceptron cannot capture.

---

**Q22.** What is the softmax function?

**A22.** The softmax function converts a vector of raw output values from a neural network into a probability distribution over predicted classes. Each output value is exponentiated and then divided by the sum of all exponentiated values, ensuring that the outputs sum to one. Log-Softmax, which computes the logarithm of softmax, is often used for the final layer in classification networks because it provides better numerical stability when combined with the negative log-likelihood loss.

---

**Q23.** What is the role of the learning rate in training?

**A23.** The learning rate controls how quickly the model moves toward the optimal parameter values during training. A learning rate that is too large can cause the optimization to overshoot and diverge, while a learning rate that is too small leads to very slow convergence. Many modern approaches use adaptive learning rate methods or learning rate schedules that adjust the rate during training.

---

**Q24.** What is stochastic gradient descent?

**A24.** Stochastic gradient descent is an optimization algorithm that updates model parameters using the gradient computed on a small random subset of the training data, called a mini-batch, rather than the entire dataset. This introduces noise into the parameter updates, which can help escape local minima and often leads to better generalization. Variants like Adam combine stochastic gradient descent with adaptive momentum to improve convergence.

---

**Q25.** What is a loss function for classification?

**A25.** For classification tasks, the cross-entropy loss function is the most commonly used. Binary cross-entropy is used for two-class problems, computing the loss based on how far the predicted probability is from the true label. For multi-class classification, categorical cross-entropy extends this to multiple classes. PyTorch provides modules like BCEWithLogitsLoss that combine a sigmoid function with binary cross-entropy for numerically stable computation.

---

**Q26.** What are optimizers in deep learning?

**A26.** Optimizers are algorithms that update the parameters of a neural network to minimize the loss function. Simple stochastic gradient descent updates parameters proportionally to the gradient, but more advanced optimizers like Adam use adaptive momentum to adjust the learning rate for each parameter individually. Other optimizers include Adadelta, which adapts the learning rate based on a running window of gradient updates, and requires minimal hyperparameter tuning.

---

## NLP and Transformers (27–36)

**Q27.** What is the transformer architecture?

**A27.** The transformer is a neural network architecture that relies entirely on the self-attention mechanism to process sequential data. Unlike recurrent neural networks, transformers can process all positions in a sequence in parallel, making them significantly more efficient to train. The transformer architecture has become the foundation for large language models and has achieved state-of-the-art results in natural language processing, computer vision, and many other domains.

---

**Q28.** What is the self-attention mechanism?

**A28.** The self-attention mechanism allows a model to weigh the importance of different positions in a sequence when computing the representation of each position. It works by computing query, key, and value vectors for each position, then using the dot product of queries and keys to determine attention weights. The inclusion of self-attention in large language models makes them significantly different and superior to approaches like high-order Markov models for language generation.

---

**Q29.** What is tokenization?

**A29.** Tokenization is the process of breaking text into smaller units called tokens, which can be words, subwords, or characters. In modern language models, subword tokenization methods like byte pair encoding are used to balance vocabulary size with the ability to represent any text. After tokenization, each token is mapped to a numerical index and then to an embedding vector that the model can process.

---

**Q30.** What are embeddings in the context of language models?

**A30.** Embeddings are dense vector representations of tokens that capture semantic meaning in a continuous vector space. Through training, tokens that appear in similar contexts develop similar embedding vectors. Embeddings are used to convert discrete tokens into numerical representations that neural networks can process, and they serve as the foundation for tasks like semantic search and retrieval augmented generation.

---

**Q31.** What is a large language model?

**A31.** A large language model is a neural network, typically based on the transformer architecture, that has been trained on a large corpus of text to predict the next token in a sequence. These models learn statistical patterns in language and can generate coherent text, answer questions, and perform a variety of natural language tasks. The quality of a language model is often measured by its perplexity, which indicates how well it predicts the next word.

---

**Q32.** What is fine-tuning in the context of language models?

**A32.** Fine-tuning is the process of taking a pretrained language model and continuing its training on a smaller, task-specific dataset. This adapts the general language understanding learned during pretraining to a specific task such as question answering or text classification. Fine-tuning typically uses a lower learning rate than pretraining to avoid catastrophically overwriting the pretrained knowledge.

---

**Q33.** What is retrieval augmented generation?

**A33.** Retrieval augmented generation, or RAG, is a technique that combines a language model with an external knowledge retrieval system. When the model needs to generate a response, it first retrieves relevant documents or passages from a knowledge base, then uses this retrieved context to generate a more accurate and grounded response. RAG helps reduce hallucination by providing the model with factual information from trusted sources.

---

**Q34.** What is prompt engineering?

**A34.** Prompt engineering is the practice of designing and refining the input text given to a language model to achieve desired outputs. The way a prompt is structured, including the instructions, examples, and context provided, can significantly affect the quality and relevance of the model's response. Techniques include zero-shot prompting, few-shot prompting with examples, and chain-of-thought prompting that encourages step-by-step reasoning.

---

**Q35.** What is text generation with language models?

**A35.** Text generation is the process of producing new text by sampling from a language model's predicted probability distribution over the next token. Parameters like temperature control the randomness of generation, with lower temperatures producing more deterministic output and higher temperatures producing more diverse text. Top-k sampling restricts the selection to the k most likely tokens, which helps prevent the model from generating unlikely or incoherent text.

---

**Q36.** What is perplexity?

**A36.** Perplexity is a metric used to evaluate language models. It measures how well the model predicts a sequence of tokens, with lower perplexity indicating better predictions. Mathematically, perplexity is the exponential of the average cross-entropy loss over the sequence. A model with a perplexity of 20 means that, on average, the model is as uncertain as if it had to choose uniformly among 20 possible next tokens.

---

## Statistics and Mathematics (37–44)

**Q37.** What is principal component analysis?

**A37.** Principal component analysis, or PCA, is a dimensionality reduction technique that transforms data into a new coordinate system where the axes, called principal components, are ordered by the amount of variance they capture. It works by computing the eigendecomposition or singular value decomposition of the data's covariance matrix. PCA is widely used for reducing the number of features while retaining the most important information in the data.

---

**Q38.** What are inner products and why are they important in machine learning?

**A38.** An inner product is an operation that takes two vectors and returns a scalar value, measuring the similarity between them. Many machine learning algorithms can be expressed solely in terms of inner products, including Linear Discriminant Analysis, Canonical Correlation Analysis, and Support Vector Machines. Given the prevalence of inner-product based learning algorithms, understanding inner products is fundamental to machine learning.

---

**Q39.** What is the difference between L1 and L2 regularization?

**A39.** L1 regularization adds the sum of the absolute values of the model parameters as a penalty to the loss function, which encourages sparsity by driving some parameters to exactly zero. L2 regularization adds the sum of the squared values of the parameters, which penalizes large weights but does not produce sparse solutions. L1 is useful for feature selection since it effectively removes irrelevant features, while L2 is more commonly used for general regularization.

---

**Q40.** What is a probability distribution?

**A40.** A probability distribution describes how the values of a random variable are spread across possible outcomes. Discrete distributions like the Bernoulli or Poisson assign probabilities to specific values, while continuous distributions like the Gaussian assign probabilities to ranges of values through a probability density function. Understanding probability distributions is fundamental because the random variation within data affects every single calculation in data science.

---

**Q41.** What is the empirical distribution function?

**A41.** The empirical distribution function estimates the cumulative distribution function of a population from observed data. For each value, it computes the proportion of observations that are less than or equal to that value. It provides a nonparametric estimate of the underlying distribution and converges to the true distribution function as the sample size grows, as guaranteed by the Glivenko-Cantelli theorem.

---

**Q42.** What is matrix decomposition?

**A42.** Matrix decomposition is the process of breaking a matrix into a product of simpler matrices that reveal its underlying structure. Common decomposition methods include eigendecomposition, singular value decomposition, and non-negative matrix factorization. These methods are applied widely in machine learning for tasks like principal component analysis, dimensionality reduction, and understanding the properties of weight matrices in neural networks.

---

**Q43.** What is the Markov property?

**A43.** The Markov property states that the future state of a process depends only on the current state and not on the sequence of events that preceded it. A Markov model uses this property to model sequential data, where the probability of the next state is conditioned on a fixed number of preceding states. Language models share some similarities with Markov models in that they use preceding context to predict the next word, though modern models like transformers use much longer contexts.

---

**Q44.** What are eigenvalues and eigenvectors?

**A44.** Eigenvalues and eigenvectors are fundamental concepts in linear algebra that describe how a linear transformation acts on a vector space. An eigenvector of a matrix is a non-zero vector that, when multiplied by the matrix, is only scaled by a constant factor called the eigenvalue. They are central to many machine learning methods including principal component analysis, spectral clustering, and analyzing the behavior of neural network weight matrices.

---

## Algorithms and Applied ML (45–50)

**Q45.** What is a decision tree?

**A45.** A decision tree is a supervised learning algorithm that makes predictions by learning a series of if-then rules from the data. At each node, the tree selects the feature and threshold that best splits the data according to a criterion like information gain or Gini impurity. Decision trees are easy to interpret and visualize, but they are prone to overfitting, which is why ensemble methods like random forests and gradient boosting are often preferred.

---

**Q46.** What is a random forest?

**A46.** A random forest is an ensemble learning method that combines multiple decision trees to make more robust predictions. Each tree is trained on a random bootstrap sample of the data and uses a random subset of features at each split, which reduces correlation between trees. Random forests and XGBoost have been applied to many types of data including time series, where the approach often taken is to treat the data as an ordinary tabular dataset.

---

**Q47.** What is support vector machine?

**A47.** A support vector machine is a supervised learning algorithm that finds the optimal hyperplane that separates data points of different classes with the maximum margin. The support vectors are the data points closest to the decision boundary. SVMs can handle non-linear classification by using kernel functions that map the data into a higher-dimensional space where a linear separator can be found. They are one of the inner-product based learning algorithms widely used in machine learning.

---

**Q48.** What is logistic regression?

**A48.** Logistic regression is a supervised learning algorithm for binary classification that models the probability of an outcome using the logistic sigmoid function. Despite its name, it is a classification method, not a regression method. The model computes a weighted sum of input features plus a bias term, then applies the sigmoid function to produce a probability between zero and one. It is trained by minimizing the binary cross-entropy loss function.

---

**Q49.** What is outlier detection?

**A49.** Outlier detection is the process of identifying data points that deviate significantly from the expected pattern in a dataset. Approaches range from simple statistical methods based on standard deviations and interquartile ranges to machine learning-based methods that learn the normal distribution of data and flag deviations. The outlier detection process involves choosing appropriate methods based on the data characteristics and the specific definition of what constitutes an anomaly.

---

**Q50.** What is time series analysis?

**A50.** Time series analysis involves methods for analyzing data points collected over time to extract meaningful statistics and identify patterns. Machine learning techniques including feed-forward neural networks, random forests, and XGBoost have been applied to time series data. The advent of deep recurrent neural networks and long short-term memory units has made it possible to build models that directly incorporate temporal dependencies, significantly advancing the field.

---

*End of Batch 1 — 50 pairs for review*
