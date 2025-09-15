from schemas import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSSTRUCTION
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain import hub
from dotenv import load_dotenv
load_dotenv()


tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

react_prompt_with_format_instruction = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSSTRUCTION,
    input_variables=["inputt", "agent_scratchpad", "tool_names", "tools"]
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm, tools=tools, prompt=react_prompt_with_format_instruction)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

chain = agent_executor | extract_output | parse_output


# summary_prompt_template = PromptTemplate.from_template(summary_template)

# #llm = ChatOpenAI(temperature=0, model="gpt-5", )
# llm = ChatOllama(model="gemma3:270m", temperature=0)

# chain = summary_prompt_template | llm
# response = chain.invoke(input={"bio": information})


def main():
    result = chain.invoke(
        input={
            "input": "search for 3 job posting for an Langchain jobs in the Pune area on linkedin for past 24hrs and list their details."
        }
    )
    print(result)


if __name__ == "__main__":
    main()
