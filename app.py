import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler

load_dotenv()

st.set_page_config(page_title="Text To Math Problem Solver ans Data Search Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

api_key = st.sidebar.text_input(label="Groq API key: ", type='password')
if not api_key:
    st.info("[-] Please add your Groq API key to continue")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key = api_key)

## initializing the tools
wiki = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki.run,
    description="A tool for searching the Internet and solving you math problem"
)

## initialize the Math Tool
math_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool(
    name="calculator",
    func=math_chain.run,
    description="A tools for answering math related questions. Only input mathematical expression need to bed provided"
)

## prompt
prompt="""
Your a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template = PromptTemplate(input_variables=['question'], template=prompt)

## create the chain to combine all the tools
chain = LLMChain(llm=llm, prompt=prompt_template)

## reasoning tool
reasoning_tool = Tool(
    name="reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)


## initialize the agent
assistant_agent = initialize_agent(
    tools=[wiki_tool, math_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)


## create session state

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {'role':'assistant', 'content':'Hi, I am a Math chatbot'}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


## function to generate the response
# def generate_response(question):
#     response = assistant_agent.invoke({'input':question})
#     return response


## let's start the interaction
question = st.text_area("Enter your question: ","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Answer"):
    if question:
        with st.spinner("Generating Response..."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            ## to generate response ,
            ## you have two options, call the generate_response function
            ## Or directly call the assistant_agent
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant', 'content':response})
            st.write("[+] Response")
            st.success(response)
    else:
        st.warning("Please enter the Question.")
