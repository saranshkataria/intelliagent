import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# Load environment variables
load_dotenv()
OCTOAI_API_TOKEN = os.environ["OCTOAI_API_TOKEN"]

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Sample resume data
resumes = {
    "resume_1": "John Doe has over 10 years of experience in software development, specializing in backend development and cloud infrastructure.",
    "resume_2": "Jane Smith is a project manager with 8 years of experience, focusing on agile methodologies and team leadership.",
    "resume_3": "Alex Johnson has 5 years of experience in software development, primarily working with frontend technologies and user interface design.",
    "resume_4": "Emily Davis is a project manager with 12 years of experience, known for her expertise in risk management and project planning.",
    "resume_5": "Michael Brown has 3 years of experience in software development, with a focus on mobile application development and cross-platform solutions.",
    "resume_6": "Sarah Wilson is a project manager with 6 years of experience, specializing in stakeholder communication and project documentation.",
    "resume_7": "David Lee has 15 years of experience in software development, with extensive knowledge in database management and performance optimization.",
    "resume_8": "Sophia Martinez is a project manager with 10 years of experience, adept at budget management and resource allocation.",
    "resume_9": "Chris Evans has 7 years of experience in software development, focusing on full-stack development and system architecture.",
    "resume_10": "Olivia Garcia is a project manager with 5 years of experience, with strengths in project scheduling and quality assurance.",
}

# Combine all resumes into a single string
combined_resumes = "\n".join(resumes.values())

# Split the combined resumes text into smaller chunks
chunk_size = 1024
chunk_overlap = 128
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)

# Split the text into chunks
splits = text_splitter.split_text(combined_resumes)

# Create Document objects from the splits
documents = [Document(page_content=split) for split in splits]

# Initialize OctoAI embeddings and LLM
llm = OctoAIEndpoint(
    model="llama-2-13b-chat-fp16",
    max_tokens=1024,
    presence_penalty=0,
    temperature=0.1,
    top_p=0.9,
)
embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/")

# Create a FAISS vector store
vector_store = FAISS.from_documents(documents, embedding=embeddings)

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Define the chat prompt templates for each step
initial_prompt_template = """You are an assistant for project planning. Use the context below to determine the number of people needed and the roles required for the given project.
Context: {context}
Task: Determine the number of people and roles needed for the project described below.
Project Description: {description}
Answer:"""

candidate_selection_prompt_template = """You are an assistant for candidate selection. Use the context below to find the best candidates that match the specified roles for the project.
Context: {context}
Task: Select the best candidates for the roles identified.
Roles: {roles}
Answer:"""

initial_prompt = ChatPromptTemplate.from_template(initial_prompt_template)
candidate_selection_prompt = ChatPromptTemplate.from_template(
    candidate_selection_prompt_template
)

# Define the initial planning chain
initial_chain = (
    {"context": retriever, "description": RunnablePassthrough()}
    | initial_prompt
    | llm
    | StrOutputParser()
)

# Define the candidate selection chain
candidate_selection_chain = (
    {"context": retriever, "roles": RunnablePassthrough()}
    | candidate_selection_prompt
    | llm
    | StrOutputParser()
)


# Route to serve the HTML page
@app.route("/")
def index():
    return render_template("index.html")


# Endpoint for answering questions
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    project_description = data.get("description")
    if not project_description:
        return jsonify({"error": "Project description is required"}), 400

    # Step 1: Determine the number of people and roles needed
    initial_response = initial_chain.invoke(project_description)
    roles_needed = initial_response.split("Answer:")[-1].strip()

    # Step 2: Select the best candidates for the identified roles
    selection_response = candidate_selection_chain.invoke(roles_needed)
    best_candidates = selection_response.split("Answer:")[-1].strip()

    return jsonify({"roles_needed": roles_needed, "best_candidates": best_candidates})


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
