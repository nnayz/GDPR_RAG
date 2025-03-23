import pandas as pd
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class GDPRAssistant:
    def __init__(self):
        # Set up the data path
        self.data_path = Path(__file__).parent.parent / "data" / "GDPR_10QA_dataset_filtered.csv"
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.llm = ChatOllama(
            model="gemma3:1b",
            temperature=0.1
        )
        
        self.setup_qa_chain()

    def setup_qa_chain(self):
        # Load and verify data
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Prepare documents
        documents = []
        for _, row in df.iterrows():
            text = f"""
            Article {row['Article Number']}: {row['Article Name']}
            Chapter {row['Chapter Number']}: {row['Chapter Name']}
            
            Content:
            {row['Content']}
            """
            documents.append(text)
        
        # Split texts
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings
        )
        
        # Create prompt template
        prompt_template = """You are a GDPR expert assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Please provide a clear and concise answer based on the GDPR articles provided in the context."""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def ask(self, question: str) -> str:
        """Ask a question and get a response"""
        return self.qa_chain.invoke(question)

    
