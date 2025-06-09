import os
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# ------------------- Configs -------------------
FAQ_FILE = "faq.txt"  # Path to business-specific FAQ file
REVIEWS_FILE = "reviews.txt"  # Path to business-specific reviews file
PERSIST_DIR = "chroma_store"
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
OUTPUT_FILE = "rag_test_results.md"

# ------------------- Util: Load & Filter Reviews -------------------
def load_reviews(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    review_blocks = content.split("\n\n")
    documents = [Document(page_content=block.strip(), metadata={"source": "reviews"}) for block in review_blocks if block.strip()]
    return documents

# ------------------- Load FAQ -------------------
def parse_faq(path: str) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    pairs = content.split("\n\n")
    documents = []
    for pair in pairs:
        if pair.startswith("Q:") and "A:" in pair:
            q, a = pair.split("A:", 1)
            question = q.replace("Q:", "").strip()
            answer = a.strip()
            full_text = f"Question: {question}\nAnswer: {answer}"
            documents.append(Document(page_content=full_text, metadata={"source": "FAQ", "priority": 2}))
    return documents

# ------------------- Prompt Template -------------------
PROMPT_TEMPLATE = """
You are a technical support assistant for a business. Provide a concise, direct answer based on the given context:
- Use precise, factual information
- Focus on the core issue
- Avoid unnecessary details or speculation
- Use only facts from the context, do not guess or assume
- Only answer questions related to the business. If the question is unrelated, state: "I can only answer questions related to the business."
- For business-related questions, use the context to provide an answer. If the context lacks information, state: "The context does not provide information on this business-related question."

If the answer relates to a problem or issue, always end with:
"If the issue persists, please contact our support for further assistance."

But if the query is a question like Which, Why, or Does, do not include the above line.

Context:
{context}

Question: {question}

Answer:
"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# ------------------- LLM Configuration -------------------
def get_openai_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=256
    )

# ------------------- Build QA Chain -------------------
def build_qa_chain():
    faq_docs = parse_faq(FAQ_FILE)
    reviews_doc = load_reviews(REVIEWS_FILE)
    all_chunks = faq_docs + reviews_doc

    # Deduplicate documents
    seen_contents = set()
    unique_chunks = [doc for doc in all_chunks if doc.page_content not in seen_contents and not seen_contents.add(doc.page_content)]

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(unique_chunks, embedding=embedding_model, persist_directory=None)

    llm = get_openai_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 50, "lambda_mult": 0.3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa

# ------------------- Test Scenarios -------------------
def get_test_scenarios():
    return {
        "Scenario 1: Direct FAQ Questions with Variations": [
            {
                "original": "What are the main features of the product or service?",
                "variations": [
                    "What are the key functionalities of the product or service?",
                    "Can you explain the primary features of the product or service?",
                    "What capabilities does the product or service offer?",
                    "How does the product or service work in terms of features?"
                ]
            },
            {
                "original": "Can I use the product or service without creating an account or logging in?",
                "variations": [
                    "Is it possible to access the product or service without registering?",
                    "Does the product or service allow usage without an account?",
                    "Can I use the product or service features without signing in?",
                    "Is registration or login required to use the product or service?"
                ]
            },
            {
                "original": "How can I manage or cancel my subscription?",
                "variations": [
                    "What’s the process to manage or terminate my subscription?",
                    "How do I handle or stop my recurring subscription?",
                    "Can you guide me on managing or canceling my subscription plan?",
                    "What steps are needed to control or end my subscription?"
                ]
            },
            {
                "original": "What are the data security and privacy controls?",
                "variations": [
                    "What measures are in place to protect data and ensure privacy?",
                    "How is my data secured and privacy maintained?",
                    "What privacy and security features are available?",
                    "Can you describe the data protection and privacy controls?"
                ]
            },
            {
                "original": "How can I restore data on a new device?",
                "variations": [
                    "What’s the process to recover data on a new device?",
                    "How do I transfer data to a new device?",
                    "Can I restore my data when switching to a new device?",
                    "What steps are needed to retrieve data on a new device?"
                ]
            }
        ],
        "Scenario 2: Merged FAQ Questions": [
            "What is the product or service, and what platforms or integrations does it support?",
            "What does the free trial include, and what subscription plans are available?",
            "Does the product or service support third-party integrations, and how are specific transactions categorized?",
            "How can I manually add data if automated tracking fails, and how are certain activities allocated?",
            "How can I use the product or service across multiple devices and share data with others?"
        ],
        "Scenario 3: Out-of-Box Questions": [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Does the product or service integrate with external platforms like cryptocurrency wallets?",
            "Can the product or service generate reports for my data?",
            "Why does the product or service crash when I try to add a new feature?",
            "Can the product or service predict future usage based on past data?",
            "What happens if I lose my backup data?"
        ]
    }

# ------------------- Run Tests and Document Results -------------------
def run_tests_and_document(qa_chain):
    scenarios = get_test_scenarios()
    results = []

    # Initialize Markdown content
    markdown = f"# RAG Model Efficiency and Consistency Report\n\n"
    markdown += "## Overview\n"
    markdown += "This report evaluates the RAG model's efficiency, consistency, and robustness for a technical support chatbot across three scenarios:\n"
    markdown += "1. **Direct FAQ Questions**: 5 FAQ questions with 4 variations each to test consistency.\n"
    markdown += "2. **Merged FAQ Questions**: 5 combined questions from 10 FAQs to test complex query handling.\n"
    markdown += "3. **Out-of-Box Questions**: 7 questions (unrelated, business-related but uncovered, edge cases) to test robustness.\n\n"
    markdown += "**Evaluation Criteria**:\n"
    markdown += "- **Efficiency**: Relevance and conciseness of responses.\n"
    markdown += "- **Consistency**: Similarity across variations.\n"
    markdown += "- **Robustness**: Handling of queries outside context.\n\n"

    # Scenario 1: Direct FAQ Questions with Variations
    markdown += "## Scenario 1: Direct FAQ Questions with Variations\n\n"
    for idx, q_set in enumerate(scenarios["Scenario 1: Direct FAQ Questions with Variations"], 1):
        markdown += f"### Question Set {idx}: {q_set['original']}\n\n"

        # Original question
        result = qa_chain.invoke(q_set['original'])
        markdown += f"**Original Question**: {q_set['original']}\n"
        markdown += "\n"
        markdown += f"**Response**: {result['result']}\n"
        markdown += "\n"

        # Variations
        for v_idx, variation in enumerate(q_set['variations'], 1):
            result = qa_chain.invoke(variation)
            markdown += f"**Variation {v_idx}**: {variation}\n"
            markdown += "\n"
            markdown += f"**Response**: {result['result']}\n"
            markdown += "\n"

    # Scenario 2: Merged FAQ Questions
    markdown += "## Scenario 2: Merged FAQ Questions\n\n"
    for idx, question in enumerate(scenarios["Scenario 2: Merged FAQ Questions"], 1):
        markdown += f"### Question {idx}: {question}\n\n"
        result = qa_chain.invoke(question)
        markdown += f"**Response**: {result['result']}\n"
        markdown += "\n"

    # Scenario 3: Out-of-Box Questions
    markdown += "## Scenario 3: Out-of-Box Questions\n\n"
    for idx, question in enumerate(scenarios["Scenario 3: Out-of-Box Questions"], 1):
        markdown += f"### Question {idx}: {question}\n\n"
        result = qa_chain.invoke(question)
        markdown += f"**Response**: {result['result']}\n"
        markdown += "\n"

    # Overall Analysis
    markdown += "## Overall Analysis\n\n"
    markdown += "- **Efficiency**: Summarize response quality across scenarios.\n"
    markdown += "- **Consistency**: Summarize consistency in Scenario 1.\n"
    markdown += "- **Robustness**: Summarize handling of Scenario 3 questions.\n"
    markdown += "- **Recommendations**: Suggest improvements (e.g., dynamic data, expanded FAQs).\n\n"

    markdown += "## Conclusion\n"
    markdown += "The RAG model was tested for efficiency, consistency, and robustness. Detailed results above provide insights into its performance and areas for enhancement.\n\n"
    markdown += "**Prepared by**: Grok 3, xAI\n"

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(markdown)

    return markdown

# ------------------- Main Execution -------------------
if __name__ == "__main__":
    # Ensure OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY in your environment variables.")

    # Build QA chain and run tests
    qa_chain = build_qa_chain()
    results = run_tests_and_document(qa_chain)
    print(f"Test results saved to {OUTPUT_FILE}")