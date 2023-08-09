#Importing required modules:
import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

#Accessing OpenAI API:
os.environ[ 'OPENAI_API_KEY' ] = apikey

#Memory:
memory = ConversationBufferMemory( input_key = 'job_title', memory_key = 'chat_history' )


#Application framework:
st.title( '🦜️🔗 GPT for simplifying and optimizing work for HR' )
prompt = st.text_input( 'Enter your job title:' ) #Creating area for entering prompt
job_des_template = PromptTemplate( input_variables = [ 'job_title' ], template = 'You are a HR. Give me the job description about {job_title}.' ) #Gives a template to a prompt
llm = OpenAI( temperature = 0.9 ) #Utilizing the LLM for our application
job_des_chain = LLMChain( llm = llm, prompt = job_des_template, verbose = True, output_key = 'job_description', memory = memory ) #Gives the prompt template to the LLM
cv_requirements_template = PromptTemplate( input_variables = [ 'job_description' ], template = 'Give me the requirements for the applicants to show to in their CV based on the {job_description}.' )
cv_requirements_chain = LLMChain( llm = llm, prompt = cv_requirements_template, verbose = True, output_key = 'cv_requirements', memory = memory )
screening_test_template = PromptTemplate( input_variables = [ 'cv_requirements' ], template = 'Based on the {cv_requirements}, ask questions for screening.' )
screening_test_chain = LLMChain( llm = llm, prompt = screening_test_template, verbose = True, output_key = 'screening_round', memory = memory )
seq_chain = SequentialChain( chains = [ job_des_chain, cv_requirements_chain, screening_test_chain ], input_variables = [ 'job_title' ], output_variables = [ 'job_description', 'cv_requirements', 'screening_round' ], verbose = True ) #Joins the chains
if prompt:
    reply = seq_chain( { 'job_title' : prompt } ) #Prompt is fed to LLM and the reply is stored in 'reply' variable.
    st.write( reply[ 'job_description' ] ) #Shows the reply of LLM on the screen
    st.write( reply[ 'cv_requirements' ] )
    st.write( reply[ 'screening_round' ] )
    with st.expander( 'Prompt History:' ):
        st.info( memory.buffer )