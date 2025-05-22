
# RAG vs RAG Comparison

Stand up two RAG pipelines; Run comparative benchmarks

## Description

This project started as an attempt to compare a Cache-augmented generation system to a Retrieval-augmented generation system.  Due to limited access to compute resources and GPUs, the CAG system couldn't be developed.  I privoted and decided to explore the variances in output I was noticing between RAG queries when sending the same query.  

I decide to stand up another RAG pipeline and programmatically pass intentinally designed queries to the systems in order to officially test their outputs.  The benchmarking script that is present allows for collection of the following metrics: number of documents retrieved from the vector stores, the length of the response from the LLM (in characters),and the total time from query till the response was returned (in seconds).  After the response was generated it was scored on 0-4 scale, with four being the highest value.

After benchmarking and scoring, an analysis script can run to generate graphic output of results, a well as an html report of the metrics for each query.

I chose to use the national vulnerability database managed by NIST and Red Canary's Atomics as two corpora used for the sake of testing.

## Getting Started

### Dependencies

- beautifulsoup4
- boto3
- botocore
- chromadb
- langchain
- langchain_community
- langchain_core
- langchain_huggingface
- langchain_text_splitters
- matplotlib
- numpy
- pandas
- Requests
- seaborn
- sentence_transformers
- streamlit
- torch
- transformers

### Script Descriptions

![image](https://github.com/user-attachments/assets/3d569282-ab8f-42aa-bb85-270f0b9b4cc1)

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

EDITING Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Kyle Simon
ksimon1@umbc.edu

## Version History

* 0.1
    * Initial Release

## License

WhiteRabbitNeo License: 
You agree not to use the Model or Derivatives of the Model:

*In any way that violates any applicable national or international law or regulation or infringes upon the lawful rights and interests of any third party; 
*For military use in any way;
*For the purpose of exploiting, harming or attempting to exploit or harm minors in any way; 
*To generate or disseminate verifiably false information and/or content with the purpose of harming others; 
*To generate or disseminate inappropriate content subject to applicable regulatory requirements;
*To generate or disseminate personal identifiable information without due authorization or for unreasonable use; 
*To defame, disparage or otherwise harass others; 
*For fully automated decision making that adversely impacts an individual’s legal rights or otherwise creates or modifies a binding, enforceable obligation; 
*For any use intended to or which has the effect of discriminating against or harming individuals or groups based on online or offline social behavior or known or predicted personal or personality characteristics; 
*To exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm; 
*For any use intended to or which has the effect of discriminating against individuals or groups based on legally protected characteristics or categories.

Outside of the WhiteRabbitNeo License, feel free to do what you'd like with this-- Open License.

## Acknowledgments

Inspiration, code snippets, etc.
* [Hugging Face in Action](https://www.manning.com/books/hugging-face-in-action) Great resource for tutorial-style learning 
* [LangChain in Action](https://www.manning.com/books/langchain-in-action)
* [Foundational work on CAG](https://github.com/hhhuang/CAG)
  
