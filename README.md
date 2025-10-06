# slm-thesis

Prototype implementation of efficient Small Language Models (SLMs) with fine-tuning, Retrieval-Augmented Generation (RAG), and quantization for resource-constrained environments - chatbot for Greek Labor Law.

All codes developed to support the thesis, specifically containing codes for fine-tuned models, creation of rag indexes for each data source, a query.py to extract the appropriate data according to the user's query, the chatbot script, as well as a flasks server together with an html file for a user-friendly interface. Finally, it also contains codes for model evaluation, both for regular model formats and for gguf formats, in order to support quantized models.


## Description

This project is a prototype implementation focusing on the development and optimization of Small Language Models (SLMs) for environments with limited computational resources. The main features include:

- Fine-tuning of pre-trained language models.
- Implementation of Retrieval-Augmented Generation (RAG) pipelines.
- Support for quantization techniques to reduce memory and computation requirements.
- Example code in Python and HTML.


## Project Structure (Indicative)

- `model/` – Code for SLM architecture and training.
- `rag/` – RAG pipeline implementations.
- `quantization/` – Scripts for model quantization.
- `web/` or `templates/` – HTML files for interface or demos.
- `README.md` – This file.

## Installation

```bash
git clone https://github.com/koxrhstos/slm-thesis.git
cd slm-thesis
pip install -r requirements.txt
```

## Usage

The repo provides scripts for training, fine-tuning, quantization, and evaluation of small language models. For more details, refer to the respective code files and comments.

## Contribution

Contributions are welcome! Open an issue or pull request with your suggestions or improvements.

---

*For any questions or issues, please contact the repository maintainer.*
