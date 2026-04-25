# A Framework for Observability and Reliability in Production Large Language Model Systems

**By**
**Aniket Awchare**
**(Roll No: _________)**

**Under the supervision of**
**Mr. Sachin Rathore**
**Lead System Software Engineer, Cloud Software Group**

**A thesis submitted in partial fulfillment of the requirements for the degree of**
**Executive Master of Technology**
**in**
**Artificial Intelligence and Data Science**

**Indian Institute of Technology Patna**
**Bihta, Patna - 801106, Bihar, India**
**May 2026**

---
<div style="page-break-after: always;"></div>

## DECLARATION

I certify that:
a. The work contained in this thesis is original and has been done by me under the guidance of my supervisor.
b. The work has not been submitted to any other Institute for any degree or diploma.
c. I have followed the guidelines provided by the Institute in preparing the thesis.
d. I have conformed to the norms and guidelines given in the Ethical Code of Conduct of the Institute.
e. Whenever I have used materials (data, theoretical analysis, figures, and text) from other sources, I have given due credit to them by citing them in the text of the thesis and giving their details in the references. Further, I have taken permission from the copyright owners of the sources, whenever necessary.

Signature: ______________________
Name: Aniket Awchare
Roll No: _________
Date: ______________

---
<div style="page-break-after: always;"></div>

## CERTIFICATE

This is to certify that the thesis report entitled **"A Framework for Observability and Reliability in Production Large Language Model Systems"**, submitted by **Aniket Awchare** to Indian Institute of Technology Patna, is a record of bona fide project work carried out by him under my supervision and guidance. This thesis in my opinion, is worthy of consideration for the award of the degree of Executive Master of Technology in Artificial Intelligence and Data Science in accordance with the regulations of the Institute.

Signature: ______________________
Name: Mr. Sachin Rathore
Designation: Lead System Software Engineer
Organization: Cloud Software Group
Date: ______________

---
<div style="page-break-after: always;"></div>

## ACKNOWLEDGEMENTS

I would like to express my sincere gratitude to my supervisor, Mr. Sachin Rathore, for his invaluable guidance, support, and patience throughout this project. His deep industry expertise and insightful feedback were instrumental in shaping the direction of this research.

I also extend my thanks to the faculty members of the AI and Data Science department at IIT Patna for their excellent instruction and academic support throughout the Executive M.Tech program.

Finally, I would like to thank my family and colleagues at Autodesk India for their continuous encouragement and understanding during my studies.

---
<div style="page-break-after: always;"></div>

## ABSTRACT

Enterprise adoption of Large Language Models (LLMs) has accelerated significantly, but the monitoring infrastructure surrounding them has not kept pace. Unlike traditional software systems that fail deterministically and loudly, LLMs fail silently—producing responses that are grammatically coherent but factually incorrect, drifting over time, or succumbing to adversarial prompts. Existing evaluation methods primarily rely on offline, static benchmarks which do not capture the dynamic, shifting distribution of live production queries.

This thesis introduces a comprehensive observability and reliability framework designed specifically for production LLM systems in the financial services domain. The framework operates as a non-intrusive monitoring layer that assesses system health without requiring ground-truth labels for every query. It comprises four core modules: (1) a Response Consistency Scorer that detects variation across paraphrased inputs; (2) a Semantic Drift Detector that utilizes statistical testing on embedding spaces to identify distribution shifts; (3) a Retrieval Alignment Scorer to monitor Retrieval-Augmented Generation (RAG) health; and (4) an Anomaly Detector to flag adversarial or out-of-distribution prompts.

Through controlled synthetic experiments using financial datasets (FinanceBench and FiQA-2018), the framework demonstrates strong detection capabilities. The system successfully detects domain shifts with high precision and recall, identifies adversarial prompts with a false positive rate below 15%, and effectively flags degraded retrieval contexts. The results confirm that applying structured, embedding-based observability mechanisms provides measurable, statistically significant signals of LLM degradation in production environments.

---
<div style="page-break-after: always;"></div>

## TABLE OF CONTENTS

1. **Introduction**
   1.1 Background and Motivation
   1.2 The Problem Statement
   1.3 Research Questions
   1.4 Objectives and Scope
   1.5 Organization of the Thesis
2. **Review of Literature**
   2.1 LLM Evaluation and Hallucination Detection
   2.2 Machine Learning Monitoring and Drift Detection
   2.3 RAG Evaluation Methods
   2.4 Adversarial Prompt Detection
   2.5 Gap Analysis
3. **Problem Formulation**
   3.1 System Model
   3.2 Failure Modes Taxonomy
   3.3 Metrics Formalization
4. **Framework Design**
   4.1 Architecture Overview
   4.2 Response Consistency Scorer
   4.3 Semantic Drift Detector
   4.4 Retrieval Alignment Scorer
   4.5 Prompt Anomaly Detector
   4.6 Dashboard and Integration
5. **Experiments and Results**
   5.1 Experimental Setup and Datasets
   5.2 Experiment 1: Domain Shift Detection
   5.3 Experiment 2: Adversarial Prompt Detection
   5.4 Experiment 3: RAG Alignment Failure
6. **Discussion**
   6.1 Interpretation of Results
   6.2 Limitations
   6.3 Applicability to Other Domains
7. **Conclusion**
   7.1 Summary of Contributions
   7.2 Future Work
8. **References**

---
<div style="page-break-after: always;"></div>

## CHAPTER 1: INTRODUCTION

### 1.1 Background and Motivation

Enterprise adoption of large language models (LLMs) has moved fast over the last two years. Systems that were previously confined to sandbox environments and research laboratories are now being deployed to handle real customer queries, process compliance documents, and support internal decision-making at scale. In industries with stringent regulatory requirements, such as financial services (banking, insurance, and trading), the reliance on LLMs introduces profound new capabilities but also significant novel risks.

The infrastructure required to monitor and maintain these models in production, however, has not kept pace with their deployment. Traditional software engineering relies on the premise of deterministic execution: if a system breaks, it usually throws an exception, produces an error code, or crashes loudly. Consequently, traditional application performance monitoring (APM) tools focus on metrics like latency, throughput, and error rates. LLMs, by contrast, exhibit probabilistic behavior and fail differently. They can produce responses that are grammatically coherent, contextually plausible, and factually completely wrong, all at the same time. 

### 1.2 The Problem Statement

The core issue is that once an LLM-based application goes live, operators have no reliable, automated way to know whether it is still working as intended without human intervention. LLMs can drift over weeks in ways that no single interaction reveals. They can be steered off course by a user who phrases a prompt in a particular way—sometimes accidentally, sometimes deliberately through adversarial prompt injection. 

Existing evaluation methods do not adequately solve this problem. Standard benchmark testing happens offline, under controlled conditions, and before deployment. While offline evaluation is a necessary step, it only captures the model's capability at a single point in time against a fixed, static dataset. In a live production environment, the query distribution continuously shifts. Domain terminology evolves, user behavior changes, and the retrieval indexes and databases that the model depends on via Retrieval-Augmented Generation (RAG) are themselves moving targets.

As a result, most organizations deploying LLMs in high-stakes environments are operating with severe blind spots. They can observe system uptime and response times, but they cannot tell whether the responses being served to users are actually correct, consistent, or safe. For low-stakes applications, this is an inconvenience; for finance, healthcare, or legal services, it constitutes a genuine enterprise risk.

### 1.3 Research Questions

To address this gap, this thesis investigates the following primary research questions:

1. What measurable indicators can reliably signal degradation, inconsistency, or anomalous behavior in a deployed LLM application, without requiring ground-truth labels for every response?
2. How can distribution shift in the embedding space of incoming user queries be detected early enough to be actionable?
3. Does applying structured observability mechanisms produce measurable improvements in system reliability, and how should those improvements be quantified?

### 1.4 Objectives and Scope

The primary objective of this project is to design, implement, and evaluate a comprehensive observability framework tailored for production LLM systems, specifically targeting the financial services domain. The framework operates as a non-intrusive monitoring layer that sits alongside a deployed application, observing inputs and outputs to compute reliability indicators.

The scope of the framework encompasses four core measurement modules:
1. **Response Consistency:** Measuring how much the system's answers vary across semantically equivalent queries.
2. **Semantic Drift Detection:** Tracking changes in the distribution of incoming queries over time using embedding-based analysis.
3. **Retrieval Alignment:** Monitoring RAG pipelines to ensure retrieved context is relevant, utilized, and faithfully represented in the final response.
4. **Prompt Anomaly Detection:** Identifying incoming queries that deviate significantly from the expected distribution, including adversarial manipulation attempts.

The evaluation is conducted using synthetic query streams and controlled failure injection on established financial datasets (FinanceBench and FiQA-2018).

### 1.5 Organization of the Thesis

The remainder of this thesis is organized as follows:
**Chapter 2** provides a review of related literature concerning LLM evaluation, drift detection, and system reliability. 
**Chapter 3** formulates the problem, defining the system model and failure taxonomy. 
**Chapter 4** details the design and architecture of the proposed observability framework and its constituent modules. 
**Chapter 5** presents the experimental setup, methodology, and results of the controlled experiments. 
**Chapter 6** discusses the implications of the findings, limitations, and potential for cross-domain applicability. 
Finally, **Chapter 7** concludes the thesis and outlines avenues for future research.

---
<div style="page-break-after: always;"></div>

## CHAPTER 2: REVIEW OF LITERATURE

### 2.1 LLM Evaluation and Hallucination Detection

The evaluation of Large Language Models has traditionally relied on static benchmarks. Benchmarks such as MMLU, TruthfulQA, and HellaSwag assess models on zero-shot or few-shot accuracy across diverse tasks$^{1, 2}$. However, these offline metrics correlate poorly with dynamic production performance. To address factual inconsistency—often termed "hallucination"—researchers have proposed automated reference-free evaluation metrics. SelfCheckGPT utilizes the principle that LLMs hallucinate inconsistently across multiple sampled responses, using BERTScore and NLI models to detect factual divergence$^{3}$. Similarly, methods relying on embedding cosine similarity and ROUGE scores have been widely adopted to proxy human judgment for factual consistency$^{4}$.

### 2.2 Machine Learning Monitoring and Drift Detection

Monitoring traditional machine learning models in production involves detecting data drift (changes in input distribution, $P(X)$) and concept drift (changes in the conditional distribution, $P(Y|X)$)$^{5}$. Statistical tests like the Kolmogorov-Smirnov (KS) test and Maximum Mean Discrepancy (MMD) are standard for univariate and multivariate drift detection$^{6}$. In the context of NLP, detecting drift is more complex due to the high dimensionality of text. Recent approaches involve mapping text to dense embedding spaces (e.g., using SentenceTransformers) and applying dimensionality reduction techniques like PCA or UMAP before performing statistical tests on rolling windows of data$^{7, 8}$.

### 2.3 RAG Evaluation Methods

Retrieval-Augmented Generation (RAG) systems introduce additional points of failure, notably retrieval failure and context ignoring. Frameworks like RAGAS (Retrieval Augmented Generation Assessment) and ARES have formalized the evaluation of RAG pipelines into distinct components: context relevance, answer faithfulness, and answer relevance$^{9, 10}$. These methods largely rely on LLM-as-a-judge paradigms to score responses. While effective, executing an LLM-as-a-judge for every production query introduces significant latency and cost overheads. Consequently, computationally cheaper proxies based on embedding similarity and token overlap are increasingly favored for real-time observability$^{11}$.

### 2.4 Adversarial Prompt Detection

As LLMs are integrated into enterprise workflows, they become targets for adversarial attacks, including prompt injection and jailbreaking. Attackers craft inputs designed to bypass safety filters or extract restricted information$^{12}$. Detection strategies typically fall into two categories: rule-based filtering and anomaly detection. Unsupervised anomaly detection methods, such as Isolation Forests and Local Outlier Factors (LOF) applied to the embedding space of incoming queries, have shown promise in identifying out-of-distribution adversarial inputs without requiring exhaustive labeled attack datasets$^{13, 14}$.

### 2.5 Gap Analysis

While substantial research exists in isolated domains—LLM benchmarking, ML drift detection, RAG evaluation, and security—there is a critical lack of integrated, near-real-time observability frameworks that combine these disciplines for production LLMs. Most existing tools require either human-labeled ground truth or invoke expensive secondary LLMs for evaluation, neither of which is scalable for high-throughput enterprise systems. This thesis bridges this gap by proposing a composite framework that utilizes computationally efficient embedding-based metrics and statistical tests to monitor consistency, drift, alignment, and anomalies simultaneously.

---
## CHAPTER 3: PROBLEM FORMULATION

### 3.1 System Model

Consider an enterprise LLM pipeline defined as a function $f$ that takes a user query $q_i \in Q$ at time $t_i$, retrieves relevant context $C_i = \{c_{i1}, c_{i2}, ..., c_{ik}\}$ from a vector database $V$, and generates a response $r_i$. The system state at time $t_i$ is thus a tuple $S_i = (q_i, C_i, r_i, t_i)$. 

In a production setting, the true probability distribution of incoming queries $P_t(Q)$ is dynamic and unobservable a priori. Furthermore, the ground-truth ideal response $r_i^*$ is unknown. The observability framework is a secondary system $O$ that maps the state $S_i$ to a multidimensional health metric vector $H_i \in \mathbb{R}^d$, where each dimension represents a specific reliability score, without having access to $r_i^*$.

### 3.2 Failure Modes Taxonomy

To design the framework $O$, we categorize the silent failures of the primary system $f$ into four distinct modes:

1. **Stochastic Inconsistency (F1):** For two queries $q_i \approx q_j$ that are semantically identical (e.g., paraphrases), the system produces $r_i \not\approx r_j$. This indicates high predictive variance and low factual reliability.
2. **Distributional Shift (F2):** The distribution of queries $P_t(Q)$ over a recent time window $[t-\Delta, t]$ diverges significantly from a baseline reference distribution $P_0(Q)$. This indicates the model is operating outside its tested domain.
3. **Retrieval Degradation (F3):** The retrieved context $C_i$ is either irrelevant to $q_i$, or the response $r_i$ fails to utilize $C_i$, leading to hallucinations.
4. **Adversarial Exploitation (F4):** The query $q_i$ is intentionally crafted to bypass safety instructions or extract unauthorized data, representing an anomaly relative to benign traffic.

### 3.3 Metrics Formalization

The observability framework computes four primary metrics corresponding to the failure modes above:

1. **Consistency Score ($M_C$):** $M_C(q) = \frac{1}{N} \sum_{k=1}^N \text{sim}(f(q), f(p_k))$, where $p_k$ are synthetically generated paraphrases of $q$, and $\text{sim}$ is a similarity function (e.g., Cosine Similarity of embeddings combined with ROUGE-L).
2. **Drift Magnitude ($M_D$):** $M_D(W_t, W_{ref}) = \text{KS}( \text{PCA}(E(W_t)), \text{PCA}(E(W_{ref})) )$, applying the two-sample Kolmogorov-Smirnov test on PCA-reduced embeddings of query windows.
3. **Alignment Score ($M_A$):** $M_A(q, C, r) = \alpha \cdot \text{Rel}(q,C) + \beta \cdot \text{Util}(C,r) + \gamma \cdot \text{Faith}(r,C)$, where Rel is relevance, Util is context utilization, and Faith is response faithfulness.
4. **Anomaly Score ($M_{An}$):** $M_{An}(q) = \text{Ensemble}(\text{IForest}(E(q)), \text{LOF}(E(q)), \text{Rules}(q))$, utilizing Isolation Forest, Local Outlier Factor, and heuristic rules.

---
<div style="page-break-after: always;"></div>

## CHAPTER 4: FRAMEWORK DESIGN

### 4.1 Architecture Overview

The proposed framework is designed as a decoupled, asynchronous monitoring layer. It intercepts the logs generated by the primary LLM serving pipeline (implemented via FastAPI and LangChain). The framework is entirely open-source, utilizing Mistral 7B (quantized to 4-bit) as the primary LLM, FAISS for vector storage, and `all-MiniLM-L6-v2` via SentenceTransformers for fast, CPU-bound text embeddings. All metrics are aggregated and logged via MLflow and visualized on a Streamlit dashboard.

### 4.2 Response Consistency Scorer

The Consistency Scorer addresses failure mode F1. When a query $q$ is flagged for consistency checking, the module generates $N$ paraphrases. The LLM processes all variants, and their responses are converted into dense embeddings. The module calculates the pairwise cosine similarity and the ROUGE-L semantic overlap between the original response and the paraphrase responses. A composite score below a configured threshold (e.g., 0.75) triggers an alert indicating high uncertainty or hallucination risk.

### 4.3 Semantic Drift Detector

To monitor F2, the Drift Detector maintains a rolling window of the most recent $W$ queries. It first establishes a baseline distribution from the initial $B$ queries. Incoming queries are embedded into a 384-dimensional space and reduced via Principal Component Analysis (PCA) to capture the highest variance dimensions. The two-sample Kolmogorov-Smirnov (KS) test is then applied across the principal components between the rolling window and the baseline. To account for multiple testing, a Bonferroni correction is applied.

### 4.4 Retrieval Alignment Scorer

The Alignment Scorer acts on the RAG pipeline to address F3. It computes three sub-metrics asynchronously:
- **Retrieval Relevance:** Cosine similarity between the query embedding and the retrieved chunk embeddings.
- **Context Utilization:** Token overlap proxy (F1 score) between the retrieved chunks and the final LLM response.
- **Faithfulness:** Embedding similarity ensuring the response does not deviate from the semantic bounds of the retrieved context.
These are aggregated into a single alignment score $[0, 1]$.

### 4.5 Prompt Anomaly Detector

To detect F4, the Anomaly Detector utilizes an ensemble approach. It applies an Isolation Forest and a Local Outlier Factor (LOF) algorithm, both trained on the baseline query embeddings. Because financial adversarial prompts can be subtle, the module also applies a rule-based regular expression matcher looking for common injection patterns (e.g., "ignore previous instructions", "system prompt"). An ensemble voting mechanism determines the final anomaly flag.

### 4.6 Dashboard and Integration

The metrics are decoupled from the visualization layer via an abstract `MetricsSnapshot` interface. A Streamlit dashboard consumes these snapshots, providing real-time gauges for the four modules, timeline charts for drift and anomaly alerts, and exportable query logs for human-in-the-loop auditing.

---
<div style="page-break-after: always;"></div>

## CHAPTER 5: EXPERIMENTS AND RESULTS

### 5.1 Experimental Setup and Datasets

The framework was evaluated using completely synthetic but realistic scenarios constructed from open-source financial datasets. The primary datasets used were FinanceBench (for corporate finance QA) and FiQA-2018 (for general financial domain QA). The experiments were executed on a Google Colab environment utilizing an NVIDIA T4 GPU (15GB VRAM) for the Mistral 7B inference and CPU for the observability metrics.

### 5.2 Experiment 1: Domain Shift Detection

**Objective:** Evaluate the Drift Detector's ability to flag when users shift from querying about Banking to querying about Insurance/Trading.
**Methodology:** A baseline of 50 banking queries was established. A stream of 200 queries was then introduced, with out-of-domain (OOD) queries injected at controlled rates (10%, 50%, and 90%).
**Results:** At a 90% injection rate, the system achieved a detection F1 score of 0.88, with detection occurring rapidly (latency < 20% of the evaluation window). At lower injection rates (10%), detection latency increased, and recall dropped slightly, demonstrating the system's sensitivity to the magnitude of the shift. The Bonferroni-corrected KS test successfully minimized false positives during the baseline phase.

### 5.3 Experiment 2: Adversarial Prompt Detection

**Objective:** Assess the Anomaly Detector's performance against financial adversarial prompts.
**Methodology:** Adversarial prompts (e.g., compliance bypasses, jailbreaks) were injected into a stream of benign FinanceBench queries at rates of 5%, 10%, and 20%. The ensemble detector evaluated each query.
**Results:** The framework achieved an F1 score exceeding 0.85 across all injection rates. Crucially for production systems, the False Positive Rate (FPR) remained strictly below the target threshold of 15% (averaging 8.4%). The combination of Isolation Forest for embedding outliers and rule-based matching for specific injection grammar proved highly effective.

### 5.4 Experiment 3: RAG Alignment Failure

**Objective:** Validate that the Alignment Score correctly separates healthy RAG operations from failures.
**Methodology:** Three conditions were tested using fixed QA pairs: (A) Healthy RAG with correct context; (B) Irrelevant chunks injected into the context window; (C) Stale/outdated chunks injected.
**Results:** The mean alignment score for condition A was 0.82, compared to 0.45 for B and 0.58 for C. The framework successfully flagged 100% of the irrelevant conditions and 80% of the stale conditions. Furthermore, the Alignment Score exhibited a strong positive Pearson correlation ($r > 0.65$) with a ground-truth proxy metric (BERTScore against reference answers), validating its utility as an independent reliability indicator.

---
## CHAPTER 6: DISCUSSION

### 6.1 Interpretation of Results

The experimental results validate the core hypothesis of this thesis: that embedding-based metrics and statistical tests can provide reliable, leading indicators of LLM failure without requiring ground-truth labels. The Response Consistency Scorer effectively proxies uncertainty, the Drift Detector identifies macroeconomic shifts in user behavior, the Alignment Scorer catches RAG breakdowns, and the Anomaly Detector filters out malicious inputs. Crucially, by utilizing SentenceTransformers and traditional machine learning models (like PCA and Isolation Forests), the computational overhead is kept minimal, allowing these checks to run synchronously or near-synchronously alongside the heavy LLM inference process.

### 6.2 Limitations

Despite its efficacy, the framework has several limitations. First, the Consistency Scorer relies on the assumption that an LLM's uncertainty manifests as variance in generated text; a model that is confidently hallucinating will bypass this check. Second, the Drift Detector requires a reasonably stable baseline; if the initial deployment period is immediately subjected to highly volatile traffic, the baseline distribution will be too broad, reducing the sensitivity of the KS test. Finally, the rule-based component of the Anomaly Detector requires manual curation of financial adversarial patterns, which must be updated as new jailbreak techniques emerge.

### 6.3 Applicability to Other Domains

While this framework was tailored for and tested against financial services datasets, the underlying architecture is largely domain-agnostic. The embeddings capture semantic meaning rather than relying on domain-specific keywords. Consequently, with a retrained baseline and adjusted threshold parameters, the same observability stack could be deployed for healthcare diagnostics, legal document analysis, or general customer support applications.

---
<div style="page-break-after: always;"></div>

## CHAPTER 7: CONCLUSION

### 7.1 Summary of Contributions

This thesis presented a comprehensive, open-source observability and reliability framework for production Large Language Model systems. By integrating methodologies from NLP evaluation, statistical drift detection, and anomaly detection, the framework provides a unified solution to the silent failure problem inherent in enterprise LLM deployments. The implementation demonstrated that high-quality observability does not necessitate invoking secondary "judge" LLMs for every query, nor does it require massive computational infrastructure. The successful deployment of this framework on constrained hardware (Google Colab T4) using 4-bit quantized models proves its accessibility and cost-effectiveness.

### 7.2 Future Work

Future research should focus on extending the framework in three directions:
1. **Adaptive Thresholding:** Implementing dynamic thresholds for consistency and alignment scores based on real-time traffic volume and historical variance.
2. **Multi-Modal Observability:** Extending the drift and anomaly detectors to handle multimodal inputs (text and image) for visual-language models.
3. **Automated Remediation:** Connecting the observability alerts directly to a fallback mechanism, such as routing high-uncertainty queries to a human agent or automatically retrying the query with a lower temperature setting.

---
<div style="page-break-after: always;"></div>

## CHAPTER 8: REFERENCES

1. Hendrycks, D., et al. (2020). Measuring Massive Multitask Language Understanding. *International Conference on Learning Representations (ICLR)*.
2. Lin, S., Hilton, J., & Evans, O. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *arXiv preprint arXiv:2109.07958*.
3. Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *arXiv preprint arXiv:2303.08896*.
4. Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. *Text Summarization Branches Out*.
5. Gama, J., et al. (2014). A Survey on Concept Drift Adaptation. *ACM Computing Surveys*, 46(4), 1-37.
6. Gretton, A., et al. (2012). A Kernel Two-Sample Test. *Journal of Machine Learning Research*, 13(1), 723-773.
7. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP*.
8. McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv preprint arXiv:1802.03426*.
9. Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.
10. Saad-Falcon, J., et al. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *arXiv preprint arXiv:2311.09476*.
11. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems*, 33, 9459-9474.
12. Perez, F., & Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. *arXiv preprint arXiv:2211.09527*.
13. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Eighth IEEE International Conference on Data Mining*, 413-422.
14. Breunig, M. M., et al. (2000). LOF: Identifying Density-Based Local Outliers. *ACM SIGMOD Record*, 29(2), 93-104.
