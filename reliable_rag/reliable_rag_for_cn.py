import os
from typing import List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

# Load environment variables from '.env' file
load_dotenv()

# llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
llm = ChatOpenAI(model="gpt-4o")
embedding_model = OpenAIEmbeddings()


# Post-processing
def format_docs(docs):
    return "\n".join(
        f"<doc{i + 1}>:\nSource:{doc.metadata['source']}\nContent:{doc.page_content}\n</doc{i + 1}>\n" for i, doc in
        enumerate(docs))[:30000]


def get_context(
        data_path: dict,
        chunk_size: int = 5000,
        chunk_overlap: int = 500,
        search_type: str = 'similarity',
        k_chunks: int = 25,
        question: str = '',
):
    loader = DirectoryLoader(data_path['path'], glob=data_path['glob'], show_progress=True, use_multithreading=False)
    documents = loader.load()
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # Increase chunk size to include more context
        chunk_overlap=chunk_overlap,  # Maintain some overlap to ensure continuity
        length_function=len,
        add_start_index=True,
    )

    def get_splits_with_context(splits):
        prompt_template = ChatPromptTemplate.from_template('''
            Please give a short succinct context to situate this chunk within the overall document for the purposes of 
            improving search retrieval of the chunk.
            Answer only with the succinct context and nothing else.
            {document}
            ''')
        parser = StrOutputParser()
        chain = prompt_template | llm | parser
        for split in splits:
            result = chain.invoke({"document": split.page_content})
            split.page_content = f'{result}\n{split.page_content}'

        return splits

    splits = text_splitter.split_documents(documents)
    splits = get_splits_with_context(splits)

    # Add to vectorstore
    # from sentence_transformers import SentenceTransformer
    # embedding_model = SentenceTransformer('emilyalsent/bio-sent2vec')
    vectorstore = Chroma.from_documents(
        documents=splits,
        collection_name="rag",
        embedding=embedding_model,
    )

    retriever = vectorstore.as_retriever(
        search_type=search_type,  # Maximal Marginal Relevance (MMR)
        search_kwargs={'k': k_chunks},  # retrieve more chunks
    )
    docs = retriever.invoke(question)

    # print(f"Source: {docs[0].metadata['source']}\n\nContent: {docs[0].page_content}\n")
    # print(f"Title: {docs[0].metadata['title']}\n\nSource: {docs[0].metadata['source']}\n\nContent: {docs[0].page_content}\n")

    return docs


def get_retrieval_grader_chain():
    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    # LLM with function call
    # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    return retrieval_grader


def get_generation_chain():
    system = '''
    You're a professional doctor and an expert medical coder at CMS, so you know the coding 
    guidelines by heart. Given the list of medical conditions and the procedures reported in the claim list, 
    analyze the text and compare it to each diagnosis code to ensure the clinical evidence and accuracy of the 
    reported medical conditions.
    '''
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Retrieved documents: \n\n <docs>{documents}</docs> \n\n User question: <question>{question}</question>"),
        ]
    )

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain


def get_hallucination_grader_chain():
    # Data model
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in 'generation' answer."""

        binary_score: str = Field(
            ...,
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    # LLM with function call
    # llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    # Prompt
    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Set of facts: \n\n <facts>{documents}</facts> \n\n LLM generation: <generation>{generation}</generation>"),
        ]
    )

    hallucination_grader = hallucination_prompt | structured_llm_grader

    return hallucination_grader


def get_doc_highlighter_chain():
    # Data model
    class HighlightDocuments(BaseModel):
        """Return the specific part of a document used for answering the question."""

        id: List[str] = Field(
            ...,
            description="List of id of docs used to answers the question"
        )

        title: List[str] = Field(
            ...,
            description="List of titles used to answers the question"
        )

        source: List[str] = Field(
            ...,
            description="List of sources used to answers the question"
        )

        segment: List[str] = Field(
            ...,
            description="List of direct segements from used documents that answers the question"
        )

    # LLM
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    # parser
    parser = PydanticOutputParser(pydantic_object=HighlightDocuments)

    # Prompt
    system = """You are an advanced assistant for document search and retrieval. You are provided with the following:
    1. A question.
    2. A generated answer based on the question.
    3. A set of documents that were referenced in generating the answer.

    Your task is to identify and extract the exact inline segments from the provided documents that directly correspond to the content used to
    generate the given answer. The extracted segments must be verbatim snippets from the documents, ensuring a word-for-word match with the text
    in the provided documents.

    Ensure that:
    - (Important) Each segment is an exact match to a part of the document and is fully contained within the document text.
    - The relevance of each segment to the generated answer is clear and directly supports the answer provided.
    - (Important) If you didn't used the specific document don't mention it.

    Used documents: <docs>{documents}</docs> \n\n User question: <question>{question}</question> \n\n Generated answer: <answer>{generation}</answer>

    <format_instruction>
    {format_instructions}
    </format_instruction>
    """

    prompt = PromptTemplate(
        template=system,
        input_variables=["documents", "question", "generation"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain
    doc_lookup = prompt | llm | parser

    return doc_lookup


if __name__ == '__main__':
    # -------------------------------------------- Retrieve documents --------------------------------------------------
    from data.prompt import question
    DATA_PATH = {'path': "data/test/", 'glob': '*.txt'}
    # DATA_PATH = {'path': "data/test2/", 'glob': '*.pdf'}
    docs = get_context(data_path=DATA_PATH,
                       chunk_size=5000,
                       chunk_overlap=500,
                       search_type='similarity',
                       k_chunks=25,
                       question=question)

    # ---------------------------------------- Grade retrieved documents -----------------------------------------------
    retrieval_grader = get_retrieval_grader_chain()
    print(f'-------------Before filtering {len(docs)}')
    graded_docs = []
    for doc in docs:
        # print(doc.page_content, '\n', '-'*50)
        res = retrieval_grader.invoke({"question": question, "document": doc.page_content})
        # print(res,'\n')
        if res.binary_score == 'yes':
            graded_docs.append(doc)
    print(f'-------------After filtering {len(graded_docs)}')

    # ------------------------------------------- Generate response ----------------------------------------------------

    rag_chain = get_generation_chain()
    # generation = rag_chain.invoke({"documents":format_docs(graded_docs), "question": question})
    generation = rag_chain.invoke({"documents": format_docs(docs), "question": question})
    print(f'--------------Answer: {generation}')

    # ------------------------------------------- Generate response ----------------------------------------------------

    hallucination_grader = get_hallucination_grader_chain()
    response = hallucination_grader.invoke({"documents": format_docs(graded_docs), "generation": generation})
    print(f'--------------Hallucinations: {response}')

    # -------------------------------------------- Highlight docs ------------------------------------------------------

    doc_lookup_chain = get_doc_highlighter_chain()
    # Run
    lookup_response = doc_lookup_chain.invoke(
        {"documents": format_docs(graded_docs), "question": question, "generation": generation})

    for id, title, source, segment in zip(lookup_response.id, lookup_response.title, lookup_response.source,
                                          lookup_response.segment):
        print(f"ID: {id}\nTitle: {title}\nSource: {source}\nText Segment: {segment}\n")
