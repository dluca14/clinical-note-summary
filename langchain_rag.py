import bs4
import dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

dotenv.load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

def load_web_docs():
    # bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    bs4_strainer = bs4.SoupStrainer("div")

    loader = WebBaseLoader(
        web_paths=(
            "https://www.icd10data.com/ICD10CM/Codes",
            "https://platform.who.int/mortality/about/list-of-causes-and-corresponding-icd-10-codes",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6724457/",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8210984/",
            "https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01428-7"
        ),
        bs_kwargs={"parse_only": bs4_strainer},
        show_progress=True,
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def load_local_docs():
    # I want to add to Chroma also the files located in the folder 'data'
    DATA_PATH = "./data"
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", show_progress=True, use_multithreading=False)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(documents)
    return splits

# all_splits = load_web_docs()
# all_splits = load_local_docs()
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory="chroma")
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="chroma")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# used one-shot inference to get results in the expected format and Prompt Engineering to teach the model how to
# prioritize urgent conditions
def get_chain():
    prompt = ChatPromptTemplate.from_template(
        '''
        You are a medical assistant helping summarize clinical notes, you provide answers based on the clinical context and
        relevant medical guidelines such as ICD-10 codes or clinical decision rules like SIRS criteria.
        Make sure you prioritize urgent conditions based on certain keywords or clinical flags
         (e.g.,"severe sepsis" if SIRS criteria are mentioned).
        Use the following pieces of retrieved context to answer the question in a concise manner.
        If you don't know the answer, just say that you don't know.

        Example clinical note:
        65-year-old female with a history of hypertension, atrial fibrillation, and chronic obstructive pulmonary disease (COPD).
        Presented to the ER with shortness of breath, wheezing, and chest tightness. Vital signs: Temp 37.8°C, HR 130 bpm,
        BP 95/55, SpO2 88% on room air. Labs: WBC 14.7, PaCO2 48 mmHg, pH 7.31, serum bicarbonate 21 mEq/L.
        Chest X-ray: Hyperinflation, no consolidation. Blood cultures pending.
        Suspected diagnosis: COPD exacerbation with possible pneumonia. Plan: Start nebulized bronchodilators, IV steroids, and antibiotics. Admit to ICU.

        Example summary:
        65-year-old female with a history of hypertension, atrial fibrillation, and COPD presents with suspected COPD exacerbation
        and possible pneumonia (shortness of breath, wheezing, chest tightness, tachycardia, hypotension, low SpO2).
        Initial labs: WBC 14.7, PaCO2 48 mmHg, pH 7.31. Plan: Admit to ICU, start nebulized bronchodilators, IV steroids,
        and antibiotics.

        Now, summarize the following clinical note in a similar format:

        Clinical note: {question}

        Context: {context}

        Summary:
        '''
    )


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def get_chain_with_sources():
    # 2. Incorporate the retriever into a question-answering chain.
    system_prompt = (
        "You are a medical assistant helping summarize clinical notes. "
        "Use relevant medical guidelines such as ICD-10 codes or clinical decision rules like SIRS criteria. "
        "Prioritize urgent conditions based on certain keywords or clinical flags "
        "(e.g., 'severe sepsis' if SIRS criteria are mentioned). "
        "Use the following pieces of retrieved context to answer the question concisely. "
        "If you don't know the answer, just say that you don't know."
        "\n\n"
        "Example clinical note:"
        "65-year-old female with a history of hypertension, atrial fibrillation, and chronic obstructive pulmonary disease (COPD)."
        "Presented to the ER with shortness of breath, wheezing, and chest tightness. Vital signs: Temp 37.8°C, HR 130 bpm,"
        "BP 95/55, SpO2 88% on room air. Labs: WBC 14.7, PaCO2 48 mmHg, pH 7.31, serum bicarbonate 21 mEq/L."
        "Chest X-ray: Hyperinflation, no consolidation. Blood cultures pending."
        "Suspected diagnosis: COPD exacerbation with possible pneumonia. Plan: Start nebulized bronchodilators, IV steroids, and antibiotics. Admit to ICU."
        "\n"
        "Example summary:"
        "65-year-old female with a history of hypertension, atrial fibrillation, and COPD presents with suspected COPD exacerbation"
        "and possible pneumonia (shortness of breath, wheezing, chest tightness, tachycardia, hypotension, low SpO2)."
        "Initial labs: WBC 14.7, PaCO2 48 mmHg, pH 7.31. Plan: Admit to ICU, start nebulized bronchodilators, IV steroids,"
        "and antibiotics."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Clinical note: {input}\n\nContext: {context}\n\nSummary:"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


if __name__ == "__main__":
    input = '''
    45-year-old male with a history of CHF, diabetes mellitus, and chronic kidney disease. Presented
    to ER with fever, tachycardia (HR: 120 bpm), and hypotension (BP: 90/60). Labs: WBC 18.3, lactate
    4.5 mmol/L, pH 7.32, serum bicarbonate 18 mEq/L. Blood cultures pending. Suspected diagnosis:
    sepsis. Plan: Start broad-spectrum antibiotics and fluids. Admit to ICU.
    '''
    expected_output = '''
    45-year-old male with a history of CHF, diabetes, and CKD presents with suspected sepsis
    (tachycardia, fever, hypotension, high lactate). 
    Initial labs: WBC 18.3, lactate 4.5 mmol/L, pH 7.32.
    Plan: Admit to ICU, start antibiotics and fluids.
    '''

    ''' There are 2 types of chains that can be used to get the answer '''
    # rag_chain = get_chain()
    # for chunk in rag_chain.stream(input):
    #     print(f'SUMMARY: {chunk}', end="", flush=True)

    rag_chain = get_chain_with_sources()
    response = rag_chain.invoke({'input': input})
    print(f'-'*50)
    print(f'INPUT: {response["input"]}')
    print(f'-'*50)
    print(f'CONTEXT: {response['context']}')
    print(f'-'*50)
    print(f'SUMMARY: {response["answer"]}')
    print(f'-'*50)
