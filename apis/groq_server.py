import asyncio
import json
import os
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config import GROQ_KEY
from groq import Groq
import logging
from langchain.agents import Tool
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function

# from utils.real_time_data import fetch_real_time_data
from openai import AsyncOpenAI, OpenAI
from langchain_core.output_parsers import PydanticToolsParser
from langchain_openai.chat_models.base import ChatOpenAI
from langchain.schema.agent import AgentFinish
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


logging.basicConfig(level=logging.INFO, filename="mylogs_groq.log")

router1 = APIRouter()
# client = Groq(api_key=GROQ_KEY)
# client = OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=GROQ_KEY,
# )

@tool
def fetch_data(question: str) -> str:
    """Fetches Trump data."""
    print(f"Tool `fetch_real_time_data` invoked with: {question}")
    return "Trump is the president of the United States in 2025."

@tool
def get_current_temperature() -> str:
    """Fetches the current temperature."""
    return "The current temperature is 25 degrees."


conversation_history = []

@router1.get("/groq")
async def make_request(input_message : str):
    

    model = ChatOpenAI(
        model="deepseek-r1-distill-llama-70b",
        api_key=GROQ_KEY,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    ).bind_tools([fetch_data, get_current_temperature])

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", "You are a helpful assistant, use tool calls to determine real-time data, and use the information to answer the user's question."),
    #         ("user", "{input}"),
    #     ]
    # )
    conversation_history.append(("User", input_message))
    
    messages = [
        SystemMessage("You are a helpful assistant, use tool calls to determine real-time data, and use the information to answer the user's question."),
        HumanMessage("{input}")
    ]
    
    message_tuples = [(type(msg).__name__.replace("Message", "").lower(), msg.content) for msg in messages]
    prompt = ChatPromptTemplate.from_messages(message_tuples)
    
    # print(f"Prompt: {prompt}")
    # print(prompt)
    
    def route(result):
        if isinstance(result, AgentFinish):
            # print(f"AgentFinish: {result}")
            return result.return_values["output"]
        else:
            print(f"Result: {result}")
            tools = {
                "fetch_data": fetch_data,
                "get_current_temperature": get_current_temperature,
            }
            
            tool_message = tools[result[0].tool].run(result[0].tool_input)
 
            # print(f"Tool: {result[0].tool}")
            return tool_message
    
    
    chain = prompt | model | OpenAIToolsAgentOutputParser() | route

    result = chain.invoke({"input": input_message})
    
    conversation_history.append(("Assistant", result))
    
    print(type(result))
    # print(result[0].tool)
    print(f"Result from chain: {result}")
    logging.info(f"Result from chain: {result}")
    return {"conversation_history": conversation_history, "result": result}
    
    
# print(fetch_real_time_data("Who is the president of the United States in 2025?"))
# asyncio.run(make_request("Who is elon musk"))



# a = fetch_real_time_data("who is the president of the United States?")
# print(a)



    # completion = client.chat.completions.create(
    #     model="deepseek-r1-distill-llama-70b",
    #     messages=[
    #         {"role": "user", "content": "Who is the president of the United States?"},
    #     ],
    #     temperature=1,
    #     max_completion_tokens=1024,
    #     top_p=1,
    #     stream=True,
    #     stop=None,
    #     # functions= [convert_to_openai_function(fetch_real_time_data)],
    #     tools=[google_search],
    #     tool_choice="auto",
    #     # functions=[convert_to_openai_function(fetch_real_time_data)]
    # )

    # # for chunk in completion:
    # #     print(chunk.choices[0].delta.content or "", end="")

    # async def stream():
    #         response = ""
    #         for chunk in completion:
    #             response += chunk.choices[0].delta.content or ""
    #             yield chunk.choices[0].delta.content or ""
    #         logging.info(f"Response completed: {response}")

    # return StreamingResponse(stream(), media_type="text/plain")
