## ✨ Introduction

A simple project showing how LLMs can be equipped with RAG capabilities using LangChain and LangGraph.

In this project, we explore how we can utilize LLMs with various tools to create a customer service chatbot that can answer customers' queries on smartphones. 

## 🤖 Agentic Workflow

![mermaid](https://i.ibb.co/YBRYyMBY/output.png)

In our workflow, we use a base LLM and support it with two tools; One for relational database retrieval and another for performing similarity search on a vector database.

Our model is only given access to a single `.csv` file containing records of smartphones and their model, price and availibility. 

During the build process, this `.csv` file is converted to a PostgreSQL database, which allows the agent to create and execute SQL queries for direct retrieval. 

At the same time, vector embeddings of the same file are also created and stored in an in-memory vector database, allowing the agent to perform similarity search if no matches are found in the relational database.

At the last step, a grader agent helps to reduce hallucinations by guiding the final LLM in generating its' response.

**Libraries used:**
- LangChain and LangGraph for designing the agentic workflow
- FastAPI for hosting an inference endpoint
- ollama for serving our LLM/embedding model
- Gradio frontend

## 🔧 Requirements

- At least 8gb of VRAM for fast inference
- ollama pre-installed
- About 1.5gb of storage (+5gb for `llama3.1:8b`)

By default, `llama3.1:8b` is used for the LLM and `nomic-embed-text:v1.5` is used for the embeddings model. This will require about 8gb of VRAM in total.

To use a different model, modify `OLLAMA_MODEL = "llama3.1:8b"` in `app/server.py`. The model being used must support tool-calling, otherwise an error will be thrown. 

Note that this project requires you to have a locally-hosted instance of ollama with the default model pulled.

## Setup and Build
1. Clone this repository
```bash
git clone https://github.com/sleepydirt/smartphone-agent.git
```
2. Run with docker
```
docker-compose up --build
```

## 🚩 Issues

1. Hallucinations
    
    The model being used (`llama3.1:8b`) is a relatively small model, making it prone to hallucinations sometimes. For example, the model might give the user details about a smartphone's colour and storage capacity, which do not exist in the data given.

2. Incorrect SQL queries

    By design, the LLM is provided the user's question and is tasked to generate a PostgreSQL query and execute it on the database. However, the model might sometimes generate incomplete/incorrect SQL queries, causing a `null` response to be returned. This might then cause the model to hallucinate.

