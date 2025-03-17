from fastapi import FastAPI
import uvicorn
import uuid
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from typing import List, Tuple, Annotated, Sequence, Literal
from typing_extensions import TypedDict
import psycopg2
import os
import json

load_dotenv()
app = FastAPI()


OLLAMA_MODEL = "llama3.1:8b"
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE")

OLLAMA_BASE_URL = "100.110.219.100:11434"

llm_json = ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json", base_url=OLLAMA_BASE_URL)
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)

ROUTER_INSTRUCTIONS = '''You are an expert at determining whether a user question requires access to a database from a smartphone shop. 

The database schema is as follows:
- id (int): The unique identifier of the smartphone. You will not use this.
- brand (str): The brand of the smartphone, (eg. 'Apple', 'Google', 'Samsung'...)
- model (str): The model of the smartphone (eg. 'iPhone 13 Pro Max', 'Galaxy S21 Ultra', 'Mi 13 Pro'...)
- price (int): The price of the smartphone in Singapore Dollars.
- stock_status (str): The stock status of the smartphone ('In Stock', 'Out of Stock')

For questions related to a smartphone's price or availibility, you should accept the user query since it requires information from the database. For unknown brands or models, you should also approve the query since it may require database access. For all other questions that do not require database access, please reject the query.

Return JSON with a single key user_query, that is 'accept' or 'reject' depending on whether the question requires database access.'''

SQL_AGENT_INSTRUCTIONS = '''You are an agent designed to interact with a PostgreSQL database.

Given an input question, create a syntactically correct PostgreSQL query to run. Your query should always begin with "SELECT * FROM smartphones WHERE" and you can add any conditions you want. Note that all string comparisons should be match case-insensitive and enclosed within single quotes.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

The database schema is as follows:
- id (int): The unique identifier of the smartphone. You will not use this.
- brand (str): The brand of the smartphone, (eg. 'Apple', 'Google', 'Samsung'...)
- model (str): The model of the smartphone (eg. 'iPhone 13 Pro Max', 'Galaxy S21 Ultra', 'Mi 13 Pro'...)
- price (int): The price of the smartphone in Singapore Dollars.
- stock_status (str): The stock status of the smartphone ('In Stock', 'Out of Stock')
'''

GRADER_INSTRUCTIONS = '''You are a grader that will be provided with data from a smartphone store's database and a user question. If the database contains information that can answer the user's question, grade it as 'yes'. Return JSON with a single key, grader_score, that is either 'yes' or 'no' to indicate whether the user's question can be answered with the data provided.'''

GRADER_PROMPT = '''Here is the data provided: \n\n {data} \n\n Here is the user question: \n\n {question}'''

RAG_PROMPT = '''You are an assistant that helps to answer users' queries about smartphone pricing and availibility.
Here is the context to use to answer the question, a database query response that contains ONLY information about the requested smartphone(s) brand, model, price, and availibility. For all other specifications, please inform the user that you do not have the information. Offer suggestions if customers type incomplete or ambiguous queries (e.g., "Did you mean iPhone 14 Pro Max or iPhone 14 Pro?").

If the requested smartphone is not present in the context, please inform the user that the smartphone is not available at the moment. Do not make up information.

Here is the context:

{context} 

Here is the user's question:

{question}

Convey all information to the user.
Answer:'''

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_result: List

# Tools
@tool
def executePostgreSQLQuery(query: str) -> List[Tuple[int, str, str, int, str]]:
    '''Executes a PostgreSQL query and returns the result as a list of tuples. Please only query the table "smartphones".
    
    Args:
        query (str): The PostgreSQL query to be executed.
        
    Returns:
        result (List[Tuple[int, str, str, int, str]]): The result of the query, a list of tuples (id, brand, model, price, stock_status).
    '''

    # Connect to database
    # print("executePostgreSQLQuery tool called!")
    conn = psycopg2.connect(database=POSTGRES_DATABASE, user=POSTGRES_USER, password=POSTGRES_PASSWORD, host="localhost", port="5432")
    # print("Database connected successfully.")
    with conn.cursor() as cursor:
        try:
            # print("Executing sql query...")
            cursor.execute(query)
            result = cursor.fetchall()
        except:
            # print("Error executing query.")
            conn.close()
            return
    # Close connection
    conn.close()
    if result:
        # print("Database query success")
        return result
    else:
        return "No result from the database"

# Edges

def grader(state) -> Literal["yes", "no"]:
    '''Determines whether the results from the database can answer the user's question. If not, suggest alternatives or inform the user that the information is not available.
    
    Args:
        state (State): The current state of the agent.
    
    Returns:
        str: A decision by the model for whether the user's question can be answered with the data provided.'''
    

    # Take the last message in the state to get the user question
    question = state["messages"][-1].content

    tool_call_result = state["tool_call_result"]

    # Format the system prompt sent to the llm with context from the database and user question
    grader_prompt_formatted = GRADER_PROMPT.format(data=tool_call_result, question=question)

    grader_response = llm_json.invoke([SystemMessage(content=GRADER_INSTRUCTIONS)] + [HumanMessage(content=grader_prompt_formatted)])
    # print("Grading the results from the database...")
    grader_score = json.loads(grader_response.content)
    
    if grader_score.get("grader_score") == "yes":
        # print("The user's question can be answered with the data provided.")
        return "yes"
    else:
        # print("The user's question cannot be answered with the data provided.")
        return "no"

def router(state) -> Literal["accept", "reject"]:
    '''Determines whether the user query requires access to the database.
    
    Args:
        state (State): The current state of the agent.
    
    Returns:
        str: Either calls the sql_agent node or generates a response directly based on the user query.'''
    
    # Take the first message in the state to get the user question
    question = state["messages"][-1]

    # Get the AIResponse containing a JSON object to indicate whether database access is accepted or rejected
    router_response = llm_json.invoke([SystemMessage(content=ROUTER_INSTRUCTIONS)] + [question])
    # print("Processing user query...")
    user_query = json.loads(router_response.content)

    if user_query.get("user_query") == "accept":
        # print("The user's query requires access to the database.")
        return "accept"
    else:
        # print("The user's query does not require access to the database.")
        return "reject"

# Nodes
def sql_agent(state):
    '''Generate a SQL query based on the user question and executes it.
    
    Args:
        state (State): The current state of the agent.
        
    Returns:
        dict: The updated state with the tool call result appended to messages.'''

    # In this case, only one tool is present, so I am hardcoding it
    # state['tools'] = StructuredTool object
    llm_with_db = llm.bind_tools([executePostgreSQLQuery])

    # Take the last message in the state to get the user question
    question = state["messages"][-1]

    # The AIResponse containing the tool call and the sql query to be performed
    sql_agent_response = llm_with_db.invoke([SystemMessage(content=SQL_AGENT_INSTRUCTIONS)] + [question])
    # print("Generating SQL query to execute...")
    # Execute the sql query and return the result
    tool_call_result = executePostgreSQLQuery.invoke(sql_agent_response.tool_calls[0])
    # print(f"Executing SQL query: {sql_agent_response.tool_calls[0].get('args').get('query')} ")

    return {"tool_call_result": tool_call_result}

def generate_response(state):
    '''Generate response to user query based on the context and user question.
    
    Args:
        state (State): The current state of the agent.
        
    Returns:
        dict: The updated state with the final response to the user query.'''

    # Take the first message in the state to get the user question

    question = state["messages"][-1].content

    # Get the tool call result from the state
    tool_call_result = state["tool_call_result"]
    
    # Format the system prompt sent to the llm with context and user question
    rag_prompt_formatted = RAG_PROMPT.format(context=tool_call_result, question=question)

    generate_response = llm.invoke([HumanMessage(content=rag_prompt_formatted)])

    return {"messages": [generate_response]}


workflow = StateGraph(State)

workflow.add_node("sql_agent", sql_agent)
workflow.add_node("generate_response", generate_response)

workflow.set_conditional_entry_point(
    router,
    {
        "accept": "sql_agent",
        "reject": END
    }
)

# should add like a max number of retries, if no go back to sql_agent
workflow.add_conditional_edges(
    "sql_agent",
    grader,
    {
        "yes": "generate_response",
        "no": "generate_response"
    }
)

checkpointer = MemorySaver()

graph = workflow.compile(checkpointer=checkpointer)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference/")
async def inference(inputs: str):
    prompt = {
    "messages": [HumanMessage(content=inputs)],
    "tool_call_result": []
    }
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # buffer = []
    # async for output, _ in graph.astream(prompt, config, stream_mode="messages"):
    #     if output.content:
    #         buffer.append(output.content)
    #         print(output.content, end='|', flush=True)
            
    # return buffer
    buffer = []
    tool_msg_received = False
    async for output, _ in graph.astream(prompt, config, stream_mode="messages"):
        if not tool_msg_received:
            buffer.append(output)
            if isinstance(output, ToolMessage):
                tool_msg_received = True
                buffer.clear()
        else:
            # After the ToolMessage, print tokens directly.
            print(output.content, end='', flush=True)
            buffer.append(output.content)
    return buffer
if __name__ == '__main__':
    uvicorn.run(app=app, port=8000)