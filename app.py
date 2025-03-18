import os
import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

allHeaders = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
}

## streamlit application
st.set_page_config(page_title = 'Langchain: Summarize Text from YT or Website URL')
st.title("Langchain: Summarize Text from Youtube or Website")
st.subheader('Summarize URL')

inputURL = st.text_input('URL', label_visibility = 'collapsed')

promptTemplate = """
Provide a Summary of the following content in no more than 1000 words:
Content: {text}
"""

prompt = PromptTemplate(template = promptTemplate, input_variables = ['text'])

if st.button('Summarize Contents from URL'):
    if not groq_api_key or not inputURL.strip():
        st.error('Please set the GROQ_API_KEY environment variable (or in .env) and provide a URL.')
    elif not validators.url(inputURL):
        st.error('Please Enter Valid URL. It should be a Youtube URL or Website URL.')
    else:
        try:
            with st.spinner('Waiting...'):
                # Initialize ChatGroq here, when both API key and URL are present
                llm = ChatGroq(model = 'gemma2-9b-it', groq_api_key=groq_api_key) 

                if "youtube.com" in inputURL:
                    loader = YoutubeLoader.from_youtube_url(inputURL, add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls = [inputURL], ssl_verify = True, headers = allHeaders)

                documents = loader.load()
                chain = load_summarize_chain(llm, chain_type = 'stuff', prompt = prompt)
                summaryOutput = chain.run(documents)

                st.success(summaryOutput)
        except Exception as e:
            st.error(f'Exception: {e}')