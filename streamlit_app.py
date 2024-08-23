import streamlit as st
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

# Streamlit UI
st.title("Intent Recognition and Response Retrieval")

# PDF Upload
uploaded_file = st.file_uploader("Upload a PDF file with transcribed calls", type="pdf")
if uploaded_file is not None:
    # Extract text from PDF
    def extract_text_from_pdf(pdf_file):
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    transcribed_text = extract_text_from_pdf(uploaded_file)

    # Preprocess the extracted text
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(filtered_tokens)

    processed_text = preprocess_text(transcribed_text)

    # Extract QA pairs
    qa_pattern = re.compile(r'(?:speaker\s*\d+:\s*)?(q\d*\.|question\s*\d*\:)(.*?)\s*(ans\.|answer\:)(.*?)\s*(?=speaker\s*\d+:|q\d*\.|question\s*\d*\:|$)', re.DOTALL | re.IGNORECASE)
    qa_pairs = re.findall(qa_pattern, processed_text)
    knowledge_base = {question.strip(): answer.strip() for _, question, _, answer in qa_pairs}

    if not knowledge_base:
        st.write("Knowledge base is empty. Check the QA extraction logic.")
    else:
        st.write(f"Knowledge base contains {len(knowledge_base)} entries.")
        st.write("Sample QA Pairs:")
        for question, answer in list(knowledge_base.items())[:5]:
            st.write(f"Q: {question}")
            st.write(f"A: {answer}")
            st.write("---")

    # Train Intent Recognition Model
    data = [
        ("I want to know my account balance", "account_balance"),
        ("How can I reset my password?", "reset_password"),
        ("Can I speak to a representative?", "speak_to_representative"),
        ("I have a problem with my order", "order_problem"),
    ]

    texts, labels = zip(*data)
    texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(texts_train, labels_train)
    predictions = model.predict(texts_test)

    st.write("Intent Recognition Model Evaluation:")
    st.write(f"Accuracy: {accuracy_score(labels_test, predictions)}")
    st.write(classification_report(labels_test, predictions))

    # Query Section
    query = st.text_input("Enter your query:")
    if query:
        def get_response(query):
            if not knowledge_base:
                return "Knowledge base is empty. Cannot provide responses."
            vectorizer = TfidfVectorizer().fit(list(knowledge_base.keys()))
            query_vector = vectorizer.transform([query])
            knowledge_vectors = vectorizer.transform(knowledge_base.keys())
            similarities = cosine_similarity(query_vector, knowledge_vectors)
            max_sim_index = similarities.argmax()

            if similarities[0, max_sim_index] > 0.3:
                most_similar_question = list(knowledge_base.keys())[max_sim_index]
                return knowledge_base[most_similar_question]
            else:
                escalate_to_expert(query)
                return "I'm forwarding your query to an expert."

        def escalate_to_expert(query):
            st.write(f"Escalating query to expert: {query}")

        response = get_response(query)
        st.write("Response:", response)
