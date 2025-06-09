RAG Technical Support Chatbot
=============================

A customizable **Retrieval-Augmented Generation (RAG)** chatbot for technical support, designed for any business. Built with **LangChain**, **Chroma**, and **OpenAI GPT-4o-mini**, it delivers accurate, context-aware responses using business-specific FAQs and reviews. Includes a robust testing framework to evaluate efficiency, consistency, and robustness.

Features
--------

*   **Contextual Responses**: Answers queries based on provided FAQs and reviews, ensuring factual and relevant replies.
    
*   **Highly Customizable**: Adapt to any business by updating faq.txt and reviews.txt.
    
*   **Comprehensive Testing**: Evaluates performance with direct, merged, and out-of-box question scenarios.
    
*   **Efficient Retrieval**: Uses Chroma vector store with MMR for precise context fetching.
    
*   **Secure Configuration**: Protects sensitive data (e.g., API keys) using environment variables.
    

Tech Stack
----------

*   **Programming Language**: Python 3.8+
    
*   **Core Libraries**:
    
    *   langchain (v0.0.352): Builds the RAG pipeline and chains LLM calls.
        
    *   langchain-community (v0.0.7): Community extensions for LangChain.
        
    *   langchain-openai (v0.0.2): Integrates with OpenAI's API.
        
    *   chromadb (v0.4.22): Vector database for document embeddings.
        
    *   huggingface\_hub (v0.20.1): Accesses multi-qa-MiniLM-L6-cos-v1 embedding model.
        
    *   python-dotenv (v1.0.0): Loads environment variables.
        
*   **LLM**: OpenAI GPT-4o-mini (configurable via API key).
    
*   **Embedding Model**: HuggingFace multi-qa-MiniLM-L6-cos-v1.
    
*   **Storage**: Chroma vector store.
    

Prerequisites
-------------

*   Python 3.8 or higher
    
*   [OpenAI API key](https://platform.openai.com/)
    
*   Git (for cloning the repository)
    

Installation
------------

1.  git clone https://github.com/anubhav-18/Rag-ChatBot.git cd Rag-ChatBot
    
2.  pip install -r requirements.txt
    
3.  **Set Up Environment Variables**:
    
    *   cp .env.example .env
        
    *   OPENAI\_API\_KEY=your\_openai\_api\_key\_here
        
4.  **Prepare FAQ and Reviews Files**:
    
    *   Q: What is your product?A: Our product is a customizable solution...Q: How do I contact support?A: Reach out via our support channels...
        
    *   Review 1: This product is easy to use and reliable.Review 2: Great customer support and fast responses.
        
5.  python rag\_chatbot.pyThis runs test scenarios and saves results to rag\_test\_results.md.
    

Customization
-------------

*   **FAQ and Reviews**: Update faq.txt and reviews.txt with your business data, following the specified formats.
    
*   **Prompt Template**: Modify PROMPT\_TEMPLATE in rag\_chatbot.py to adjust response tone or instructions.
    
*   **Test Scenarios**: Edit get\_test\_scenarios in rag\_chatbot.py to include business-specific queries.
    
*   **Embedding Model**: Change EMBEDDING\_MODEL in the config to use a different HuggingFace model.
    
*   **LLM Settings**: Adjust model, temperature, or max\_tokens in get\_openai\_llm for custom response behavior.
    

Testing
-------

The chatbot includes a testing framework with three scenarios:

1.  **Direct FAQ Questions**: 5 FAQs with 4 variations each to test consistency.
    
2.  **Merged FAQ Questions**: 5 combined questions to test complex query handling.
    
3.  **Out-of-Box Questions**: 7 unrelated or edge-case questions to test robustness.
    

Run tests with:

`python rag_chatbot.py`

Results are saved to rag\_test\_results.md.

Deployment
----------

To deploy as a web service:

1.  Integrate the QA chain with a web framework (e.g., [FastAPI](https://fastapi.tiangolo.com/) or [Flask](https://flask.palletsprojects.com/)).
    
2.  Host on a cloud platform (e.g., [Heroku](https://www.heroku.com/), [AWS](https://aws.amazon.com/), or [Vercel](https://vercel.com/)).
    
3.  Persist the Chroma vector store by setting PERSIST\_DIR to a stable location.
    
4.  Secure the OpenAI API key using environment variables or a secret manager.
    

Contributing
------------

Contributions are welcome! To contribute:

1.  Fork the repository.
    
2.  git checkout -b feature/your-feature
    
3.  git commit -m "Add your feature"
    
4.  git push origin feature/your-feature
    
5.  Open a pull request.
    
License
-------

This project is licensed under the [MIT License].

Support
-------

For issues or questions, open a [GitHub Issue](https://github.com/anubhav-18/Rag-ChatBot/issues).
