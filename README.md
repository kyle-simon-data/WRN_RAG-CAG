
# RAG vs RAG Comparison

Stand up two RAG pipelines; Run comparative benchmarks

## Description

This project started as an attempt to compare a Cache-augmented generation system to a Retrieval-augmented generation system.  Due to limited access to compute resources and GPUs, the CAG system couldn't be developed.  I privoted and decided to explore the variances in output I was noticing between RAG queries when sending the same query.  

I decide to stand up another RAG pipeline and programmatically pass intentinally designed queries to the systems in order to officially test their outputs.  The benchmarking script that is present allows for collection of the following metrics: number of documents retrieved from the vector stores, the length of the response from the LLM (in characters),and the total time from query till the response was returned (in seconds).  After the response was generated it was scored on 0-4 scale, with four being the highest value.

After benchmarking and scoring, an analysis script can run to generate graphic output of results, a well as an html report of the metrics for each query.

I chose to use the national vulnerability database managed by NIST and Red Canary's Atomics as two corpora used for the sake of testing.

## Getting Started

### Dependencies

EDITING
* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* 

### Executing program

EDITING
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

STILL EDITING THIS -- This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

STILL EDITING THIS
Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)