# clinical-note-summary
LLM with RAG for summarizing clinical notes

### Some observations related to my work:
- used ChatGPT model because of time constraints, although I've found some other models that might be more suitable (https://nlp.johnsnowlabs.com/2023/07/06/clinical_notes_qa_base_en.html)
- used RAG system to augment the llm with medical data, avoiding hallucinations and understanding medical abbreviations. There are 2 ways of adding documents, locally or online resources.
- used prompt engineering to prioritize urgent conditions and instruct the model on what is expected.
- used one-shot inference to provide the model with an example of how I would expect the answer to be.
- created a sample ocr script for preprocessing the scanned docs (if provided with an example document I can further develop the script).