# All the langchains
from langchain.agents import (
    Tool,
    AgentExecutor,
    load_tools,
    initialize_agent,
    ZeroShotAgent,
)
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import DeepLake
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import (
    RetrievalQA,
)
from langchain.chains.router import MultiRetrievalQAChain
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)

import openai
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

load_dotenv()


class OLDAGENT:
    def __init__(self, memory, agent_chain):
        self.memory = memory
        self.agent = agent_chain
        self.embeddings = OpenAIEmbeddings()

    def create_chatagent(seed_memory=None):
        chat = ChatOpenAI(
            streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0
        )
        search = GoogleSearchAPIWrapper()
        embeddings = OpenAIEmbeddings()

        # Analyst retriever
        analyst_vs_path = "./Deeplake/snkl_helper/research_data/"
        analyst_vs = DeepLake(dataset_path=analyst_vs_path, embedding=embeddings)
        analyst_retriever = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=analyst_vs.as_retriever(),
            return_source_documents=False,
        )

        # Snorkel retriever
        snorkel_vs_path = "./Deeplake/snkl_helper/snorkel_data/"
        snorkel_vs = DeepLake(dataset_path=snorkel_vs_path, embedding=embeddings)
        snorkel_retriever = RetrievalQA.from_chain_type(
            llm=chat,
            chain_type="stuff",
            retriever=snorkel_vs.as_retriever(),
            return_source_documents=False,
        )

        tools = [
            Tool(
                name="Analyst reports",
                func=analyst_retriever.run,
                description="useful for all questions related to industry analysts reports from companies like Gartner",
            ),
            Tool(
                name="Snorkel Knowledge",
                func=snorkel_retriever.run,
                description="useful for all questions related to analysts",
            ),
            Tool(
                name="Current Search",
                func=search.run,
                description="useful for all question that asks about current events",
            ),
        ]

        # memory = (
        #     seed_memory
        #     if seed_memory is not None
        #     else ConversationBufferMemory(memory_key="chat_history")
        # )

        # agent_chain = initialize_agent(
        #     tools,
        #     chat,
        #     agent="conversational-react-description",
        #     verbose=True,
        #     memory=memory,
        # )

        return ChatAgent()
