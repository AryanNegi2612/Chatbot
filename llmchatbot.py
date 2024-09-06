import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load the Hugging Face model (GPT-Neo 2.7B or smaller for free usage)
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Function to extract text from the uploaded PDF file
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Function to embed text chunks using Sentence Transformer
def embed_text(text_chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text_chunks)

# Function to find the most relevant chunk using cosine similarity
def find_similar_chunk(query, text_chunks, embeddings, threshold=0.5):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    best_match_index = similarities.argmax()
    best_score = similarities[best_match_index]
    
    # If the score is below the threshold, escalate to an expert
    if best_score < threshold:
        return None, best_score
    
    return text_chunks[best_match_index], best_score

# Function to generate the answer using GPT-Neo (Hugging Face)
def get_answer_from_llm(context, question):
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = generator(input_text, max_length=200, do_sample=True, temperature=0.7)
    return response[0]['generated_text'].strip()

# Function to escalate the question to an expert (logging for simplicity)
def escalate_to_expert(question):
    # Log the escalated question in a file
    with open('escalation_log.txt', 'a') as file:
        file.write(f"Escalated Question: {question}\n")
    # Notify the user that their question has been escalated
    return "Your question has been escalated to an expert. They will get back to you shortly."

# Streamlit app layout and logic
def main():
    st.title("Customer Support Chatbot")

    # Step 1: Upload the PDF
    uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_pdf is not None:
        # Step 2: Extract text from PDF
        text = extract_text_from_pdf(uploaded_pdf)
        # Split text into chunks for better searchability
        text_chunks = text.split('\n\n')  # Simple chunking by paragraphs
        # Embed the text chunks for similarity search
        embeddings = embed_text(text_chunks)

    # Step 3: Take the user's question
    user_question = st.text_input("Ask a question:")
    if user_question and uploaded_pdf is not None:
        # Step 4: Find the relevant chunk from the PDF
        relevant_chunk, score = find_similar_chunk(user_question, text_chunks, embeddings)

        # Step 5: If relevant chunk is found and score is high enough
        if relevant_chunk:
            # Get answer from GPT-Neo (Hugging Face)
            answer = get_answer_from_llm(relevant_chunk, user_question)
            st.write("Answer:", answer)
            st.write(f"Confidence Score: {score:.2f}")
        else:
            # If no relevant data is found, escalate to expert
            escalation_message = escalate_to_expert(user_question)
            st.write(escalation_message)

# Run the Streamlit app
if __name__ == "__main__":
    main()
