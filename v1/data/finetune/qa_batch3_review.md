# Q&A Pairs — Batch 3 (Review)

Mark each pair: **APPROVE**, **EDIT**, or **REJECT**

---

## LLMs and Generative AI (101–118)

**Q101.** How are large language models trained?

**A101.** Large language models are trained through a pretraining phase where the model learns to predict the next token in a sequence from a large corpus of text. The model processes massive amounts of text data, learning statistical patterns in language including grammar, facts, and reasoning patterns. After pretraining, the model can be fine-tuned on specific tasks or aligned with human preferences through techniques like reinforcement learning from human feedback.

---

**Q102.** What is the difference between autoregressive and autoencoder language models?

**A102.** Autoregressive language models generate text one token at a time from left to right, predicting each next token based on the preceding context. GPT is an example of an autoregressive model. Autoencoder models like BERT are trained to reconstruct masked tokens from bidirectional context, making them strong at understanding tasks like classification. The choice between these architectures depends on whether the primary task is generation or understanding.

---

**Q103.** What is a foundation model?

**A103.** A foundation model is a large model trained on broad data that can be adapted to a wide range of downstream tasks. These models serve as a foundation because their general-purpose representations, learned during pretraining on diverse data, can be specialized through fine-tuning or prompting. Foundation models have transformed the development of AI applications because they reduce the need to train task-specific models from scratch.

---

**Q104.** What is hallucination in language models?

**A104.** Hallucination occurs when a language model generates text that is fluent and confident but factually incorrect or unsupported by the input context. This is a fundamental challenge because language models learn to produce plausible-sounding text based on statistical patterns rather than grounded factual knowledge. Retrieval augmented generation is one approach to reducing hallucination by providing the model with relevant factual information from trusted sources.

---

**Q105.** What is a knowledge graph?

**A105.** A knowledge graph is a structured representation of information that stores entities and their relationships as a network of nodes and edges. Entities represent real-world concepts like people, places, or technical terms, while edges represent relationships between them. Knowledge graphs are used with large language models to provide structured factual knowledge that can improve the accuracy of generated responses and reduce hallucination.

---

**Q106.** What is an AI agent?

**A106.** An AI agent is a system that uses a large language model to autonomously plan and execute multi-step tasks. Unlike a simple chatbot that responds to individual prompts, an agent can break down complex goals into subtasks, use external tools, access databases, and make decisions about which actions to take next. The rise of AI agents represents a shift from passive language model interaction to active, goal-directed autonomous systems.

---

**Q107.** What is the difference between zero-shot and few-shot learning?

**A107.** Zero-shot learning refers to a model's ability to perform a task without any task-specific examples, relying solely on the knowledge acquired during pretraining and the task description in the prompt. Few-shot learning provides a small number of examples in the prompt to demonstrate the desired input-output pattern. Large language models have shown remarkable few-shot learning abilities, often matching the performance of models that were specifically fine-tuned on much larger datasets.

---

**Q108.** What is model evaluation for AI systems?

**A108.** Model evaluation for AI systems involves measuring how well a model performs its intended task using appropriate metrics and test data. For language models, evaluation includes both automated metrics like perplexity and human evaluation of output quality. Evaluation methodology must carefully consider what the system is intended to do, what failure modes exist, and how to measure performance in a way that reflects real-world usage.

---

**Q109.** What is chain-of-thought prompting?

**A109.** Chain-of-thought prompting is a technique where the language model is encouraged to show its reasoning process step by step before arriving at a final answer. By breaking complex problems into intermediate steps, the model can solve problems that require multi-step reasoning more accurately. This approach has been shown to significantly improve performance on mathematical, logical, and commonsense reasoning tasks.

---

**Q110.** What are word embeddings?

**A110.** Word embeddings are dense vector representations of words in a continuous vector space where semantically similar words are located near each other. They are learned from large text corpora by training models that predict words from their context or vice versa. Word embeddings capture semantic relationships, so that vector arithmetic can express analogies: the vector for king minus man plus woman approximates the vector for queen.

---

**Q111.** What is attention in neural networks?

**A111.** Attention is a mechanism that allows a neural network to focus on the most relevant parts of the input when producing each element of the output. It computes a weighted sum of input representations, where the weights indicate how much attention each input position receives. Self-attention, used in transformers, computes these weights based on the similarity between query and key vectors derived from the same input sequence.

---

**Q112.** What is multi-head attention?

**A112.** Multi-head attention runs multiple attention computations in parallel, each with different learned projection matrices. Each attention head can learn to focus on different aspects of the input, such as syntactic relationships, semantic similarity, or positional patterns. The outputs of all heads are concatenated and projected to produce the final result. This allows the model to jointly attend to information from different representation subspaces.

---

**Q113.** What is positional encoding in transformers?

**A113.** Positional encoding provides the transformer model with information about the position of each token in the sequence. Since the self-attention mechanism treats all positions equally, without positional information the model would have no way to distinguish between different orderings of the same tokens. Common approaches include sinusoidal positional encodings that use fixed mathematical functions and learned positional embeddings that are trained along with the model.

---

**Q114.** What is the feed-forward network in a transformer?

**A114.** Each transformer layer contains a feed-forward network that is applied to each position independently. It typically consists of two linear transformations with a nonlinear activation function like GELU or ReLU in between. The feed-forward network expands the representation to a higher dimension, applies the nonlinearity, and then projects it back to the model dimension. This component adds the capacity for the model to learn complex transformations at each layer.

---

**Q115.** What is layer normalization?

**A115.** Layer normalization normalizes the activations across the features of each individual sample, rather than across the batch as in batch normalization. It computes the mean and variance across all features for each sample independently, then normalizes and applies learned scale and shift parameters. Layer normalization is the standard choice in transformer architectures because it works well with variable-length sequences and is independent of batch size.

---

**Q116.** What is the difference between encoder and decoder in transformers?

**A116.** The encoder processes the input sequence and produces a contextualized representation of each token, using bidirectional self-attention that can attend to all positions. The decoder generates output tokens one at a time, using causal self-attention that can only attend to previous positions to prevent information leakage from future tokens. Some models use both components, while others use only an encoder like BERT or only a decoder like GPT.

---

**Q117.** What is beam search in text generation?

**A117.** Beam search is a decoding strategy that maintains multiple candidate sequences simultaneously during text generation. At each step, it expands each candidate by considering the most probable next tokens and keeps only the top candidates based on their cumulative probability. Compared to greedy decoding, which always selects the single most probable token, beam search explores more of the search space and often produces higher-quality outputs.

---

**Q118.** What is the role of temperature in text generation?

**A118.** Temperature is a parameter that controls the randomness of text generation by scaling the logits before applying the softmax function. A temperature of 1.0 uses the model's raw probabilities, while lower temperatures make the distribution sharper, favoring the most likely tokens and producing more deterministic output. Higher temperatures flatten the distribution, increasing diversity but also the risk of generating less coherent text.

---

## Algorithms and Data Structures (119–130)

**Q119.** What is binary search?

**A119.** Binary search is an efficient algorithm for finding an element in a sorted list. It works by repeatedly dividing the search space in half: comparing the target value to the middle element and eliminating the half that cannot contain the target. This gives a time complexity of O(log n), which is much faster than the O(n) time required to check every element in a linear search.

---

**Q120.** What is quicksort?

**A120.** Quicksort is a divide-and-conquer sorting algorithm that works by selecting a pivot element and partitioning the array into elements less than and greater than the pivot. It then recursively sorts the two partitions. On average, quicksort runs in O(n log n) time, making it one of the fastest general-purpose sorting algorithms. However, poor pivot choices can degrade performance to O(n squared) in the worst case.

---

**Q121.** What is a hash table?

**A121.** A hash table is a data structure that stores key-value pairs and provides fast lookup, insertion, and deletion operations. It uses a hash function to map keys to positions in an array, achieving average-case O(1) time complexity for these operations. Hash tables are one of the most important and widely used data structures in programming, used to implement dictionaries, sets, and caches.

---

**Q122.** What is breadth-first search?

**A122.** Breadth-first search is a graph traversal algorithm that explores all nodes at the current depth level before moving to nodes at the next depth level. It uses a queue data structure to keep track of which nodes to visit next. Breadth-first search finds the shortest path between two nodes in an unweighted graph and is used in many applications including network routing and social network analysis.

---

**Q123.** What is depth-first search?

**A123.** Depth-first search is a graph traversal algorithm that explores as far as possible along each branch before backtracking. It uses a stack data structure, which can be implemented using recursion. Depth-first search is used for tasks like topological sorting, finding connected components, and solving maze problems. It uses less memory than breadth-first search because it only needs to store the nodes along the current path.

---

**Q124.** What is dynamic programming?

**A124.** Dynamic programming is a technique for solving complex problems by breaking them into simpler overlapping subproblems and storing the results to avoid redundant computation. Each subproblem is solved only once and its result is stored in a table for future reference. This approach converts exponential time algorithms into polynomial time by eliminating repeated calculations, and is used for optimization problems like shortest paths and sequence alignment.

---

**Q125.** What is the difference between a graph and a tree?

**A125.** A tree is a special type of graph that is connected and has no cycles, meaning there is exactly one path between any two nodes. A general graph can have cycles, multiple paths between nodes, and disconnected components. Trees have a hierarchical structure with a root node and parent-child relationships, while graphs represent more general relationships between entities. Both are fundamental data structures in computer science.

---

**Q126.** What is recursion?

**A126.** Recursion is a programming technique where a function calls itself to solve a problem by breaking it into smaller instances of the same problem. Every recursive function needs a base case that stops the recursion and a recursive case that reduces the problem toward the base case. While recursion provides elegant solutions for problems like tree traversal and divide-and-conquer algorithms, it can be less efficient than iterative solutions due to the overhead of function calls.

---

**Q127.** What is the Big O notation?

**A127.** Big O notation describes the upper bound of an algorithm's time or space complexity as a function of the input size. It characterizes how the algorithm's resource requirements grow as the input gets larger. Common complexities include O(1) for constant time, O(log n) for logarithmic, O(n) for linear, O(n log n) for linearithmic, and O(n squared) for quadratic. This notation helps compare algorithms and predict their behavior on large inputs.

---

**Q128.** What is a greedy algorithm?

**A128.** A greedy algorithm makes the locally optimal choice at each step with the hope of finding a global optimum. It selects the best available option at each decision point without looking ahead or reconsidering previous choices. Greedy algorithms work well for certain problems like Huffman coding and minimum spanning trees where the locally optimal choice leads to a globally optimal solution, but they do not guarantee optimal solutions for all problems.

---

**Q129.** What is Dijkstra's algorithm?

**A129.** Dijkstra's algorithm finds the shortest path from a source node to all other nodes in a weighted graph with non-negative edge weights. It maintains a set of visited nodes and repeatedly selects the unvisited node with the smallest known distance, then updates the distances to its neighbors. The algorithm uses a priority queue for efficient selection and runs in O(V log V + E) time where V is the number of vertices and E is the number of edges.

---

**Q130.** What is Huffman coding?

**A130.** Huffman coding is a lossless data compression algorithm that assigns variable-length binary codes to characters based on their frequency of occurrence. More frequent characters receive shorter codes, while less frequent characters receive longer codes. The algorithm builds a binary tree from the bottom up, starting with the least frequent characters. This approach produces an optimal prefix-free code that minimizes the total number of bits needed to represent the data.

---

## Data Engineering (131–142)

**Q131.** What is a data pipeline?

**A131.** A data pipeline is an automated process that extracts data from various sources, transforms it into a suitable format, and loads it into a destination system for analysis or model training. Pipelines handle tasks like data cleaning, feature computation, and quality validation at each stage. Well-designed pipelines are reproducible, testable, and can be scheduled to run at regular intervals.

---

**Q132.** What is Apache Spark?

**A132.** Apache Spark is a distributed computing framework designed for large-scale data processing. It processes data in parallel across a cluster of machines, making it capable of handling datasets that are too large for a single computer. Spark supports batch processing, streaming, machine learning, and SQL queries through a unified API. PySpark provides a Python interface to Spark, making it accessible to data scientists.

---

**Q133.** What is a feature store?

**A133.** A feature store is a centralized repository for storing, managing, and serving features used in machine learning models. It ensures that the same feature definitions are used consistently across training and inference, preventing training-serving skew. Feature stores handle feature computation, versioning, and serving at both batch and real-time scales, making it easier to reuse features across different models and teams.

---

**Q134.** What is ETL?

**A134.** ETL stands for Extract, Transform, Load, which describes the three phases of a data pipeline. The extract phase pulls data from source systems like databases, APIs, or files. The transform phase cleans, validates, and reshapes the data for the target use case. The load phase writes the transformed data to the destination system such as a data warehouse or feature store. Modern variations include ELT, where raw data is loaded first and transformed in the destination system.

---

**Q135.** What is a data warehouse?

**A135.** A data warehouse is a centralized repository that stores structured data from multiple sources, optimized for analytical queries and reporting. Unlike operational databases that are designed for fast transactional processing, data warehouses are designed for complex queries across large volumes of historical data. They use schemas like star and snowflake to organize data into fact and dimension tables for efficient analytical processing.

---

**Q136.** What is a data lakehouse?

**A136.** A data lakehouse is an architecture that combines the low-cost storage of a data lake with the data management and performance features of a data warehouse. It stores data in open formats on cloud object storage while providing ACID transactions, schema enforcement, and indexing capabilities. Platforms like Databricks implement the lakehouse architecture using Delta Lake, which adds a transaction layer on top of cloud storage.

---

**Q137.** What is data versioning?

**A137.** Data versioning tracks changes to datasets over time, similar to how version control systems like Git track changes to code. It allows teams to reproduce experiments by linking specific model versions to the exact data they were trained on. Data versioning is essential for debugging model issues, meeting regulatory requirements, and understanding how changes in training data affect model performance.

---

**Q138.** What is streaming data processing?

**A138.** Streaming data processing handles data in real time as it arrives, rather than processing it in large batches. This is essential for applications like monitoring, fraud detection, and real-time recommendation systems where timely processing matters. Frameworks like Apache Spark Streaming and Apache Kafka enable processing of continuous data streams with low latency while maintaining the ability to join streaming data with historical data.

---

**Q139.** What is cloud object storage?

**A139.** Cloud object storage is a scalable and cost-effective way to store large amounts of unstructured data in the cloud. Services like Amazon S3 and Azure Blob Storage store data as objects, each with a unique identifier, metadata, and the data itself. Object storage is the foundation for data lakes and lakehouse architectures because it can store any type of data at virtually unlimited scale with high durability and availability.

---

**Q140.** What is MLOps?

**A140.** MLOps combines machine learning with DevOps practices to automate and streamline the lifecycle of machine learning models in production. It covers model training, testing, deployment, monitoring, and retraining. MLOps addresses challenges specific to machine learning systems, such as data drift, model degradation, and the need to track experiments and reproduce results. The goal is to make the process of deploying and maintaining ML models as reliable as traditional software deployment.

---

**Q141.** What is model monitoring in production?

**A141.** Model monitoring tracks the performance and behavior of deployed machine learning models to detect issues before they impact users. It includes monitoring for data drift, where the distribution of incoming data changes from what the model was trained on, and concept drift, where the relationship between inputs and outputs changes. Monitoring systems alert teams when model performance degrades, triggering retraining or other corrective actions.

---

**Q142.** What is experiment tracking?

**A142.** Experiment tracking records the parameters, metrics, code, and data associated with each machine learning experiment. This allows data scientists to compare different approaches, reproduce results, and understand which changes led to improvements. Tools for experiment tracking log hyperparameters, training curves, evaluation metrics, and model artifacts, making it possible to systematically search for the best model configuration.

---

## Bayesian Methods (143–150)

**Q143.** What is Bayesian inference?

**A143.** Bayesian inference is a method of statistical inference that updates the probability of a hypothesis as new evidence is observed. It starts with a prior distribution that represents initial beliefs about the parameters, then combines the prior with the likelihood of the observed data using Bayes' theorem to produce a posterior distribution. The posterior represents the updated beliefs about the parameters after seeing the data.

---

**Q144.** What is a prior distribution?

**A144.** A prior distribution represents the initial beliefs or knowledge about a parameter before observing any data. Informative priors incorporate specific domain knowledge, while uninformative priors aim to let the data speak for itself. The choice of prior can significantly influence the posterior, especially when data is limited. As more data is observed, the influence of the prior diminishes and the posterior becomes increasingly determined by the data.

---

**Q145.** What is Markov Chain Monte Carlo?

**A145.** Markov Chain Monte Carlo, or MCMC, is a class of algorithms for sampling from probability distributions that are difficult to compute directly. It works by constructing a Markov chain whose stationary distribution equals the target distribution, typically the posterior distribution in Bayesian inference. By running the chain for many steps, the samples approximate the target distribution and can be used to estimate quantities like posterior means and credible intervals.

---

**Q146.** What is a hierarchical model?

**A146.** A hierarchical model is a Bayesian model where the parameters themselves have distributions governed by higher-level parameters, called hyperparameters. This structure allows information to be shared across groups while still allowing group-specific variations. For example, in modeling student test scores across schools, a hierarchical model can learn both the overall distribution of school performance and the specific characteristics of each school.

---

**Q147.** What is the difference between frequentist and Bayesian statistics?

**A147.** Frequentist statistics interprets probability as the long-run frequency of events and treats model parameters as fixed but unknown values. Bayesian statistics treats parameters as random variables with probability distributions that are updated as data is observed. Frequentists use methods like confidence intervals and p-values, while Bayesians use posterior distributions and credible intervals. Each framework has strengths depending on the problem and available data.

---

**Q148.** What is probabilistic programming?

**A148.** Probabilistic programming allows users to define complex statistical models using a programming language and automatically perform inference on those models. Instead of manually deriving and implementing inference algorithms, the user specifies the model structure and the framework handles the computation of posterior distributions. Libraries like PyMC provide a Python interface for building and fitting Bayesian models using MCMC and variational inference.

---

**Q149.** What is variational inference?

**A149.** Variational inference is an alternative to MCMC for approximating posterior distributions. It frames the inference problem as an optimization problem by finding a simpler distribution that is as close as possible to the true posterior, measured by a divergence metric. Variational inference is typically faster than MCMC because it uses gradient-based optimization, making it suitable for large datasets and complex models where MCMC would be too slow.

---

**Q150.** What is posterior predictive checking?

**A150.** Posterior predictive checking is a technique for evaluating whether a Bayesian model adequately captures the patterns in the observed data. It works by generating simulated datasets from the posterior predictive distribution and comparing them to the actual data. If the model is a good fit, the simulated datasets should look similar to the real data. Systematic discrepancies between simulated and observed data indicate aspects of the data that the model fails to capture.

---

*End of Batch 3 — Q101–Q150 for review*
