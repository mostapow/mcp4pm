from sqlalchemy import create_engine, text
import pandas as pd
from mcp.server.fastmcp import FastMCP
from config import DATABASE_URL, SERVER_CONFIG
from typing_extensions import TypedDict
from langchain_community.utilities import SQLDatabase
import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import pm4py
from langgraph.graph import START, StateGraph

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

db = SQLDatabase.from_uri(DATABASE_URL)
engine = create_engine(DATABASE_URL)

system_message = """
Given an input question, create a syntactically correct {dialect} SELECT query to
run to help find the answer. You must ONLY generate SELECT queries - no INSERT, UPDATE, DELETE, or other DML/DDL statements are allowed.

Unless the user specifies in his question a specific number of examples they wish to obtain, 
always limit your query to at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Some column names have colons in their names, so while writing queries you have to put quotation marks around them.

Only use the following tables:
{table_info}

IMPORTANT: If the user's question requires modifying data (INSERT, UPDATE, DELETE) or creating/modifying database objects (CREATE, ALTER, DROP),
respond with an error message explaining that only SELECT queries are supported.
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

os.environ["ANTHROPIC_API_KEY"] = SERVER_CONFIG.anthropic_api_key

llm = init_chat_model("claude-3-5-sonnet-latest", model_provider="anthropic")


mcp = FastMCP("sql-generator-server")


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}



# Tool: Answer a user question about the database using LLM-powered SQL generation and execution
@mcp.tool("answer_database_question")
async def answer_database_question(question: str) -> dict:

    # Step 1: Generate SQL query
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 10,
        "table_info": db.get_table_info(),
        "input": question,
    })
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    sql_query = result["query"]

    # Step 2: Execute SQL query
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    sql_result = execute_query_tool.invoke(sql_query)

    # Step 3: Generate final answer
    answer_prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {question}\n'
        f'SQL Query: {sql_query}\n'
        f'SQL Result: {sql_result}'
    )
    response = llm.invoke(answer_prompt)
    return {
        "question": question,
        "sql_query": sql_query,
        "sql_result": sql_result,
        "answer": response.content
    }


if __name__ == "__main__":
    # Initialize and run the server
    #pass
    mcp.run(transport='stdio')