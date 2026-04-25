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
*(Note: Remaining chapters 3 to 8 will follow the exact same formatting structure)*
