import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import ConfigurableField
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")

st.set_page_config(page_title="Temperature Tuner Agent", layout="wide")

st.title("Temperature Tuner: Deterministic vs. Creative")
st.markdown("""
Adjust the **Temperature** and **Max Tokens** of the LLM on-the-fly without re-initializing the model.
- **Low Temp `(0.1)`**: Deterministic, factual, good for data extraction or trading bots.
- **High Temp `(0.9)`**: Creative, random, good for storytelling.
""")

# Sidebar settings n
st.sidebar.header("Agent Settings")

mode = st.sidebar.radio("Mode", ["Single Run", "Compare (Deterministic vs Creative)"])

if mode == "Single Run":
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
else:
    st.sidebar.info("Comparison Mode: Running the same chain with Low (0.1) and High (0.9) temperature.")

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_SERVER)

# Load vector store (with explicit deserialization flag)
vectorstore = FAISS.load_local(
    "logs_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Prompt for retrieval QA
prompt = PromptTemplate.from_template(
    "Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Models
mistral = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)

retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | mistral
    | StrOutputParser()
)

# 1. Define LLM with default settings
llm = OllamaLLM(model="lfm2:latest", temperature=0.5, base_url=OLLAMA_SERVER)

# 2. Make fields configurable
# This allows us to override 'temperature' and 'num_predict' at runtime
configurable_llm = llm.configurable_fields(
    temperature=ConfigurableField(id="llm_temperature", name="LLM Temperature", description="The temperature of the LLM"),
)

input_text = st.text_input("Enter your request:", value="Write a short haiku about coding.")

if st.button("Run Agent"):
    with st.spinner("Running..."):
        if mode == "Single Run":
            # 4. Invoke with config containing the slider values
            response = qa_chain.invoke(
                input_text,
                config={"configurable": {"llm_temperature": temperature}}
            )
            # need add ollama token
            
            st.subheader("Output")
            st.write(response)
            
            st.divider()
            st.subheader("Configuration Used")
            st.json({
                "model": "lfm2:latest",
                "temperature": temperature
            })
            st.code(f"""
# The Magic Line
chain.invoke(..., config={{
    "configurable": {{
        "llm_temperature": {temperature}
    }}
}})
            """)
        else:
            # Comparison Mode
            col1, col2 = st.columns(2)
            
            # Run Deterministic
            with col1:
                st.subheader("🔵 Deterministic (Temp 0.1)")
                with st.spinner("Running Deterministic..."):
                    response_det = qa_chain.invoke(
                        input_text,
                        config={"configurable": {"llm_temperature": 0.1}}
                    )
                    st.write(response_det)
                    st.caption("Config: `temperature=0.1`")

            # Run Creative
            with col2:
                st.subheader("🔴 Creative (Temp 0.9)")
                with st.spinner("Running Creative..."):
                    response_creative = qa_chain.invoke(
                        input_text,
                        config={"configurable": {"llm_temperature": 0.9}}
                    )
                    st.write(response_creative)
                    st.caption("Config: `temperature=0.9`")
            
            st.divider()
            st.subheader("💡 equivalent code")
            st.code("""
# One object, two behaviors!
chain.invoke(..., config={"configurable": {"llm_temperature": 0.1}})
chain.invoke(..., config={"configurable": {"llm_temperature": 0.9}})
            """, language="python")

