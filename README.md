A simple project showing how LLMs can be equipped with RAG capabilities using LangChain and LangGraph

## Features

![mermaid](https://ibb.co/DfVs8yfs)

The LLM is equipped with two tools; One for database retrieval with PostgreSQL and another for performing similarity search on a vector database.

As a chatbot meant to answer customers' queries about smartphones, it is only given access to a single `.csv` file containing records of smartphones and their model, price and availibility. 

During build, this `.csv` file is converted to a PostgreSQL database, which allows the agent to execute SQL queries for direct retrieval. Alternatively, vector embeddings of the same file are also created, allowing the agent to perform similarity search on a vector database if no matches are found in the relational database.

At the last step, a grader agent helps to reduce hallucinations by guiding the final LLM in generating its' response.

**Libraries used**
- LangChain and LangGraph for designing the agentic workflow
- FastAPI for hosting an inference endpoint
- Gradio frontend

## Hardware requirements

- At least 8gb of VRAM for fast inference
- About 13gb of storage

By default, `llama3.1:8b` is used for the LLM and `nomic-embed-text:v1.5` is used for the embeddings model. This will require about 8gb of VRAM in total.

To use a different model, modify `OLLAMA_MODEL = "llama3.1:8b"` in `app/server.py`.

## Setup and Build
1. Clone this repository
```bash
git clone https://github.com/sleepydirt/smartphone-agent.git
```
2. Run with docker
```
docker-compose up --build
```

## Issues

