---
license: apache-2.0
---

## Dataset Description:
TechQA-RAG-Eval is a reduced version of the original TechQA ([IBM’s GitHub Page](https://github.com/ibm/techqa), [HuggingFace](https://huggingface.co/datasets/PrimeQA/TechQA/tree/main))  dataset specifically for evaluating Retrieval-Augmented Generation (RAG) systems. The dataset consists of technical support questions and their answers, sourced from real IBM developer forums where acceptable answers included links to reference technical documentation.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
05/05/2025

## License/Terms of Use: 
Apache-2.0

## Intended Usage:
TechQA-RAG-Eval is particularly well-suited for:
Benchmarking RAG system performance on technical domain queries
Evaluating information retrieval systems in technical support contexts
Testing natural language understanding and generation for technical support applications


## Dataset Characterization
| Aspect | Details |
|-----------------------|----------------|
| Data Collection Method| Automated |
| Labeling Method | Not Applicable |

## Dataset Format
The dataset is composed of .txt and .json files.


## Dataset Quantification
| Metric | Value |
|--------------------|--------------|
| Record Count | 908 question/answer pairs |
| Feature Count | 5  |
| Features |  ['id', 'question', 'answer', 'is_impossible', 'contexts'] | 
| Data Storage Size | 46 MB (.zip)|


## Reference(s):
TechQA: https://github.com/ibm/techqa 

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.   

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
