# Q&A Pairs — Batch 4 (Review)

Mark each pair: **APPROVE**, **EDIT**, or **REJECT**

---

## NLP Tasks and Techniques (151–164)

**Q151.** What is named entity recognition?

**A151.** Named entity recognition is a natural language processing task that identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, and dates. It is typically framed as a sequence labeling problem where each token is assigned a label. Modern approaches use pretrained language models fine-tuned on entity-labeled datasets to achieve high accuracy across different domains.

---

**Q152.** What is sentiment analysis?

**A152.** Sentiment analysis is the task of determining the emotional tone or opinion expressed in a piece of text. It can be binary, classifying text as positive or negative, or more fine-grained with multiple sentiment categories. Approaches range from simple methods using sentiment lexicons to deep learning models that learn to capture nuanced expressions of opinion from large annotated datasets.

---

**Q153.** What is text classification?

**A153.** Text classification assigns predefined labels to text documents based on their content. Applications include spam detection, topic categorization, and language identification. Modern approaches use pretrained language models that are fine-tuned on labeled examples for the target classification task. The model processes the input text and produces a probability distribution over the possible labels.

---

**Q154.** What is machine translation?

**A154.** Machine translation is the task of automatically translating text from one language to another. Early approaches used rule-based and statistical methods, but modern neural machine translation uses sequence-to-sequence models with attention mechanisms. The transformer architecture has dramatically improved translation quality by enabling the model to learn complex alignments between source and target languages through self-attention.

---

**Q155.** What is text summarization?

**A155.** Text summarization generates a shorter version of a document while preserving its key information. Extractive summarization selects the most important sentences from the original text, while abstractive summarization generates new sentences that convey the main points. Large language models have shown strong performance on abstractive summarization by leveraging their ability to understand and rephrase content.

---

**Q156.** What is question answering?

**A156.** Question answering is the task of automatically generating an answer to a question, either from a given context passage or from the model's learned knowledge. Extractive question answering identifies the span of text in a passage that answers the question, while generative question answering produces a free-form answer. Retrieval augmented generation combines both approaches by first retrieving relevant passages and then generating an answer from them.

---

**Q157.** What are stop words in text processing?

**A157.** Stop words are common words like articles, prepositions, and conjunctions that occur frequently in text but carry little semantic meaning. Removing stop words is a common preprocessing step in natural language processing that reduces the vocabulary size and can improve the performance of models that rely on word frequency. However, modern transformer-based models typically process text without stop word removal, as these words can carry important syntactic information.

---

**Q158.** What is TF-IDF?

**A158.** TF-IDF, or Term Frequency-Inverse Document Frequency, is a numerical statistic that reflects how important a word is to a document within a collection. The term frequency measures how often a word appears in a document, while the inverse document frequency measures how rare the word is across all documents. Words that appear frequently in a specific document but rarely in the overall collection receive higher TF-IDF scores, making this metric useful for information retrieval and text mining.

---

**Q159.** What is sequence-to-sequence modeling?

**A159.** Sequence-to-sequence modeling maps an input sequence to an output sequence of potentially different length. It uses an encoder to process the input sequence into a fixed representation and a decoder to generate the output sequence one token at a time. This architecture is the foundation for tasks like machine translation, text summarization, and dialogue generation. The addition of attention mechanisms allows the decoder to focus on relevant parts of the input at each generation step.

---

**Q160.** What is byte pair encoding?

**A160.** Byte pair encoding is a subword tokenization algorithm that builds a vocabulary by iteratively merging the most frequent pairs of adjacent tokens. Starting from individual characters, it repeatedly finds and merges the most common pair until a desired vocabulary size is reached. This approach balances the ability to represent any text, including rare or unseen words, with the efficiency of a manageable vocabulary size. It is widely used in modern language models.

---

**Q161.** What is a language model's vocabulary?

**A161.** A language model's vocabulary is the set of tokens that the model can recognize and generate. It is determined during tokenizer training and typically includes common words, subword units, individual characters, and special tokens. The vocabulary size affects the model's ability to represent text efficiently: a larger vocabulary reduces the number of tokens needed per sentence but increases the size of the embedding and output layers.

---

**Q162.** What is attention masking in transformers?

**A162.** Attention masking prevents certain positions from being attended to during the self-attention computation. In causal language models used for text generation, a causal mask ensures that each token can only attend to previous tokens, preventing information from flowing backward in time. Padding masks are used to ignore padding tokens in variable-length sequences so they do not influence the attention computation.

---

**Q163.** What is the difference between greedy and sampling-based decoding?

**A163.** Greedy decoding always selects the token with the highest probability at each step, producing deterministic output that tends to be repetitive. Sampling-based decoding randomly selects tokens from the probability distribution, introducing diversity at the cost of potential incoherence. Top-k sampling restricts the selection to the k most likely tokens, while nucleus sampling selects from the smallest set of tokens whose cumulative probability exceeds a threshold, balancing quality and diversity.

---

**Q164.** What is the context window of a language model?

**A164.** The context window is the maximum number of tokens that a language model can process in a single forward pass. Tokens beyond the context window cannot be directly attended to by the model. The context window size is determined by the model architecture and the positional encoding scheme used. Longer context windows allow the model to consider more surrounding text when making predictions, but they increase memory usage and computational cost quadratically with the attention mechanism.

---

## Model Architecture and Design (165–178)

**Q165.** What is the encoder-decoder architecture?

**A165.** The encoder-decoder architecture is a neural network design where the encoder processes the input and produces a compressed representation, and the decoder uses that representation to generate the output. In sequence tasks, the encoder reads the entire input sequence and the decoder generates the output sequence one step at a time. This architecture is used in machine translation, summarization, and image captioning.

---

**Q166.** What is a skip connection?

**A166.** A skip connection, also called a residual connection, allows the input to a layer to bypass the layer and be added directly to the layer's output. This creates a shortcut path for gradients during backpropagation, making it easier to train very deep networks. Skip connections are a fundamental component of both residual networks in computer vision and transformer architectures in natural language processing.

---

**Q167.** What is the GELU activation function?

**A167.** The GELU, or Gaussian Error Linear Unit, is an activation function that smoothly combines the properties of ReLU with a probabilistic gating mechanism. Unlike ReLU which has a hard cutoff at zero, GELU smoothly scales values near zero based on a Gaussian distribution. GELU has become the standard activation function in transformer architectures because it provides better empirical performance than ReLU for language modeling tasks.

---

**Q168.** What is weight tying in language models?

**A168.** Weight tying is a technique where the same weight matrix is shared between the input embedding layer and the output prediction layer of a language model. Since both layers map between token indices and hidden representations, sharing weights reduces the total number of parameters without harming performance. This technique is widely used in modern language models because it improves sample efficiency and often leads to better generalization.

---

**Q169.** What is pre-norm versus post-norm in transformers?

**A169.** Pre-norm and post-norm refer to the placement of layer normalization within a transformer block. In post-norm, normalization is applied after the residual connection, which was the original design. In pre-norm, normalization is applied before the attention or feed-forward sublayer. Pre-norm has become the preferred choice for training stability because it produces more uniform gradient magnitudes across layers, making it easier to train deep models.

---

**Q170.** What is the difference between model depth and width?

**A170.** Model depth refers to the number of layers in a neural network, while width refers to the dimensionality of the representations within each layer. Deeper models can learn more abstract hierarchical features, while wider models have more capacity at each level of abstraction. In transformers, depth is the number of transformer blocks and width is the model dimension. Both affect the total parameter count and the model's capacity to learn complex patterns.

---

**Q171.** What is an autoencoder?

**A171.** An autoencoder is a neural network trained to reconstruct its input by passing it through a bottleneck layer with fewer dimensions. The encoder compresses the input into a lower-dimensional representation, and the decoder attempts to reconstruct the original input from this compressed form. Autoencoders learn useful representations of the data that can be used for dimensionality reduction, denoising, and anomaly detection.

---

**Q172.** What is a generative adversarial network?

**A172.** A generative adversarial network consists of two neural networks trained in competition: a generator that creates synthetic data and a discriminator that tries to distinguish between real and generated data. Through this adversarial training process, the generator learns to produce increasingly realistic outputs. GANs have been particularly successful for generating images, though they can be challenging to train due to mode collapse and training instability.

---

**Q173.** What is a variational autoencoder?

**A173.** A variational autoencoder extends the standard autoencoder by learning a probabilistic mapping to a structured latent space. Instead of mapping each input to a single point, the encoder outputs the parameters of a probability distribution in the latent space. Samples from this distribution are decoded to generate new data. This probabilistic framework enables controlled generation and smooth interpolation between data points.

---

**Q174.** What is model distillation?

**A174.** Model distillation is a technique for transferring the knowledge from a large, complex model to a smaller, more efficient one. The smaller student model is trained to mimic the output probability distribution of the larger teacher model, not just the hard labels. This allows the student to capture the nuanced relationships learned by the teacher while being much faster and cheaper to run at inference time.

---

**Q175.** What is quantization of neural networks?

**A175.** Quantization reduces the precision of neural network weights and activations from 32-bit floating point to lower precision formats like 16-bit or 8-bit integers. This significantly reduces memory usage and can speed up inference on hardware that supports low-precision computation. Post-training quantization applies after training is complete, while quantization-aware training incorporates quantization effects during training to minimize accuracy loss.

---

**Q176.** What is the U-Net architecture?

**A176.** U-Net is a convolutional neural network architecture designed for image segmentation, named for its U-shaped structure. It consists of a contracting encoder path that captures context through successive downsampling, and an expanding decoder path that enables precise localization through upsampling. Skip connections between corresponding encoder and decoder layers allow the network to combine high-resolution features with deep semantic features, making it particularly effective for medical image segmentation.

---

**Q177.** What is an LSTM?

**A177.** Long Short-Term Memory is a type of recurrent neural network architecture designed to learn long-term dependencies in sequential data. It uses a gating mechanism with forget, input, and output gates that control the flow of information through the cell. The cell state acts as a memory that can carry information across many time steps, solving the vanishing gradient problem that prevents standard recurrent neural networks from learning long-range dependencies.

---

**Q178.** What is the difference between RNN, LSTM, and GRU?

**A178.** A standard recurrent neural network processes sequences by maintaining a hidden state that is updated at each time step, but it struggles with long-range dependencies due to the vanishing gradient problem. LSTM addresses this with a cell state and three gates that control information flow, enabling it to remember information over long sequences. GRU, or Gated Recurrent Unit, simplifies the LSTM by combining the forget and input gates into a single update gate, achieving similar performance with fewer parameters.

---

## Cloud and Production ML (179–190)

**Q179.** What is model deployment?

**A179.** Model deployment is the process of making a trained machine learning model available to serve predictions in a production environment. This involves packaging the model, setting up an inference server, defining an API for receiving requests and returning predictions, and ensuring the system can handle the expected load. Deployment considerations include latency requirements, throughput, cost, and the ability to update models without downtime.

---

**Q180.** What is A/B testing for machine learning models?

**A180.** A/B testing compares two versions of a machine learning model by randomly assigning users to see predictions from either the current model or a new candidate model. By measuring the impact on business metrics like conversion rate or user engagement, teams can make data-driven decisions about whether to deploy the new model. This approach provides a controlled way to evaluate model changes in production before rolling them out to all users.

---

**Q181.** What is data drift?

**A181.** Data drift occurs when the statistical properties of the input data change over time relative to the data the model was trained on. This can happen due to changes in user behavior, seasonal patterns, or shifts in the data collection process. When data drift occurs, the model's predictions may become less accurate because the patterns it learned during training no longer match the current data distribution.

---

**Q182.** What is a model registry?

**A182.** A model registry is a centralized system for storing, versioning, and managing machine learning models throughout their lifecycle. It tracks model versions along with their metadata, performance metrics, and lineage information. The registry enables teams to promote models through stages like development, staging, and production, and provides a single source of truth for which model version is currently deployed.

---

**Q183.** What is containerization for ML deployment?

**A183.** Containerization packages a machine learning model along with all its dependencies, libraries, and runtime environment into a self-contained unit called a container. Docker is the most widely used container technology. Containers ensure that the model runs identically in development, testing, and production environments, eliminating the common problem of models working differently across different machines or configurations.

---

**Q184.** What is batch inference versus real-time inference?

**A184.** Batch inference processes a large number of predictions at once, typically on a schedule, and stores the results for later use. Real-time inference generates predictions on demand for individual requests with low latency requirements. Batch inference is more efficient and cost-effective for use cases where predictions can be precomputed, while real-time inference is necessary when predictions depend on the most recent data or user input.

---

**Q185.** What is model explainability?

**A185.** Model explainability refers to the ability to understand and interpret how a machine learning model makes its predictions. Techniques include feature importance scores that indicate which inputs most influence the output, attention visualization that shows what parts of the input the model focuses on, and methods like SHAP and LIME that explain individual predictions. Explainability is important for building trust, debugging models, and meeting regulatory requirements.

---

**Q186.** What is federated learning?

**A186.** Federated learning is a training approach where multiple devices or organizations collaboratively train a model without sharing their raw data. Each participant trains a local model on their own data and shares only the model updates, which are aggregated to improve the global model. This approach preserves data privacy and is particularly useful in healthcare and mobile applications where data cannot leave the device or institution.

---

**Q187.** What is the difference between edge and cloud inference?

**A187.** Cloud inference runs the model on remote servers, offering access to powerful hardware but requiring network connectivity and introducing latency. Edge inference runs the model directly on the end device, such as a phone or microcontroller, providing low latency and offline capability but with limited computational resources. TinyML focuses on deploying machine learning models on microcontrollers with very limited memory and processing power.

---

**Q188.** What is continuous training?

**A188.** Continuous training automatically retrains machine learning models when new data becomes available or when performance degrades. This ensures that models remain accurate as the underlying data distribution changes over time. A continuous training pipeline monitors model performance, triggers retraining based on predefined criteria, validates the new model, and automatically deploys it if it meets quality thresholds.

---

**Q189.** What is training-serving skew?

**A189.** Training-serving skew occurs when the data or features used during model training differ from those available at inference time. This can happen when feature engineering code is implemented differently in the training pipeline and the serving system, or when features that were available during training are not available in real time. Feature stores help prevent training-serving skew by ensuring consistent feature computation across both environments.

---

**Q190.** What is responsible AI?

**A190.** Responsible AI encompasses the practices and principles for developing AI systems that are fair, transparent, and aligned with human values. It addresses challenges like algorithmic bias, where models may discriminate against certain groups due to biased training data, and the need for transparency in how AI systems make decisions. Ethical AI challenges include balancing model performance with fairness, ensuring privacy, and maintaining human oversight of autonomous systems.

---

## Specialized Topics (191–200)

**Q191.** What is web scraping?

**A191.** Web scraping is the automated process of extracting data from websites. It involves sending HTTP requests to web servers, parsing the HTML or JSON responses, and extracting the desired information. Common tools include libraries that parse HTML structure and navigate the document tree. Web scraping is used to collect training data, monitor prices, and aggregate information from multiple sources.

---

**Q192.** What is vector search?

**A192.** Vector search finds the most similar items in a collection by comparing their vector representations. Each item is converted to a dense vector using an embedding model, and similarity is measured using metrics like cosine similarity or Euclidean distance. Vector databases are optimized for fast nearest-neighbor search over millions of vectors, making them essential for applications like semantic search, recommendation systems, and retrieval augmented generation.

---

**Q193.** What is a graph neural network?

**A193.** A graph neural network is a type of neural network designed to operate on graph-structured data, where entities are represented as nodes and their relationships as edges. It learns node representations by aggregating information from neighboring nodes through message passing. Graph neural networks can perform tasks like node classification, link prediction, and graph classification, and they are used in applications ranging from social network analysis to molecular property prediction.

---

**Q194.** What is multimodal learning?

**A194.** Multimodal learning involves building models that can process and relate information from multiple data modalities such as text, images, audio, and video. The challenge lies in learning shared representations that capture the relationships between different modalities. Approaches include early fusion, which combines modalities at the input level, and late fusion, which processes each modality separately before combining their representations.

---

**Q195.** What is curriculum learning?

**A195.** Curriculum learning is a training strategy that presents examples to the model in a meaningful order, typically starting with simpler examples and gradually increasing difficulty. This mirrors how humans learn and can lead to faster convergence and better generalization compared to random data presentation. The difficulty of examples can be measured by loss values, confidence scores, or predefined heuristics based on the data characteristics.

---

**Q196.** What is label smoothing?

**A196.** Label smoothing is a regularization technique that softens the target labels during training by replacing the hard one-hot encoded targets with a mixture of the original labels and a uniform distribution. Instead of training the model to predict a probability of exactly one for the correct class, label smoothing encourages the model to be less confident, which can improve generalization and calibration of the predicted probabilities.

---

**Q197.** What is knowledge distillation for language models?

**A197.** Knowledge distillation for language models trains a smaller student model to replicate the behavior of a larger teacher model. The student learns from the teacher's full output probability distribution over the vocabulary, which contains richer information than the hard labels alone. This process transfers the teacher's learned patterns about word relationships and contextual understanding to the smaller model, enabling deployment in resource-constrained environments.

---

**Q198.** What is contrastive learning?

**A198.** Contrastive learning is a self-supervised learning approach that trains models by comparing pairs of examples. The model learns to produce similar representations for positive pairs, which are different views of the same example, and dissimilar representations for negative pairs, which are different examples. This approach has been highly successful for learning visual and text representations without requiring labeled data.

---

**Q199.** What is instruction tuning?

**A199.** Instruction tuning fine-tunes a pretrained language model on a diverse set of tasks described through natural language instructions. Instead of training separate models for each task, a single model learns to follow various instructions like translating text, summarizing documents, or answering questions. This improves the model's ability to generalize to new tasks described by instructions it has not seen during training.

---

**Q200.** What is the mixture of experts architecture?

**A200.** The mixture of experts architecture uses multiple specialized sub-networks, called experts, with a gating mechanism that routes each input to the most relevant experts. Only a subset of experts is activated for any given input, allowing the model to have a very large total parameter count while keeping the computational cost per input manageable. This approach enables training models with trillions of parameters that would be impractical if every parameter were used for every input.

---

*End of Batch 4 — Q151–Q200 for review*
