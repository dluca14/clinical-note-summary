import math
import openai
import os
from docx import Document  # For creating Word files


def generate_analysis_for_chunk(chunk):
    """
    Generate an analysis using the GPT-4 API for a given chunk of text.
    """
    prompt = f"""
    I need you to analyze the following document text and provide clinical insights:
    {chunk}

    Please be detailed and quote from the provided text to support your analysis. Make sure to accurately extract the clinical 
    information and analyze it for evidence supporting medical conditions and procedures.
    Provide your response in this format:
    - Patient Information (Age, Gender, Length of Stay)
    - Past Medical History
    - Presenting problems/complaints
    - Vitals
    - Pertinent labs, testing, and results
    - Pertinent treatment
    - Documentation to support clinical validation
    - Rationale for diagnosis/procedures
    - Conclusion
    """

    try:
        response = openai.chat.completions.create(
        # response = openai.ChatCompletion.create(
            model="gpt-4",  # Use GPT-4 model
            messages=[
                {"role": "system", "content": "You are a medical expert analyzing clinical data for validation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,
            temperature=0.7  # Set creativity/variability in responses
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error querying GPT-4: {e}"


def save_to_word(file_name, result):
    """
    Save the result text to a Word document.
    """
    doc = Document()
    doc.add_paragraph(result)
    output_path = f'./{file_name}.docx'
    doc.save(output_path)
    print(f"Result saved to {output_path}")


def chunk_document(text, max_tokens=1000):
    """
    Break the document into smaller chunks to fit within token limits.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def analyze_document(file_path, output_name):
    """
    Process the document by chunking and analyzing each section through GPT-4.
    """
    # Load the document
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as file:
            document_text = file.read()

    # Chunk the document to avoid exceeding token limits
    chunks = chunk_document(document_text, max_tokens=1000)

    result = ""
    # Process each chunk using GPT-4
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}...")
        chunk_result = generate_analysis_for_chunk(chunk)
        result += f"Response for chunk {idx + 1}:\n{chunk_result}\n\n"

    # Save the final result to a Word document
    # save_to_word(output_name, result)
    print(result)


# Main Execution
# Example usage: Analyze a document and save results to a Word file
file_path = ''  # Path to the document you want to analyze
output_name = ''  # Name for the output Word file

# Analyze the document and save results
analyze_document(file_path, output_name)
