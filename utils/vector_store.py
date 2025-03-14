from typing import List, Optional
import os
from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader
)
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class DocumentProcessor:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings()
        self.index_path = Path("vector_store")
        self.index_path.mkdir(parents=True, exist_ok=True)
        
    def load_documents(self) -> List[Document]:
        """Load documents from the docs directory."""
        documents = []
        
        # Configure loaders for different file types
        loaders = {
            "*.txt": DirectoryLoader(
                self.docs_dir,
                glob="**/*.txt",
                loader_cls=TextLoader
            ),
            "*.pdf": DirectoryLoader(
                self.docs_dir,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            ),
            "*.ppt*": DirectoryLoader(
                self.docs_dir,
                glob="**/*.ppt*",
                loader_cls=UnstructuredPowerPointLoader
            ),
            "*.doc*": DirectoryLoader(
                self.docs_dir,
                glob="**/*.doc*",  # This will match both .doc and .docx
                loader_cls=UnstructuredWordDocumentLoader
            )
        }
        
        # Load all documents
        for glob_pattern, loader in loaders.items():
            try:
                docs = loader.load()
                print(f"Loaded {len(docs)} documents from {glob_pattern}")
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {glob_pattern}: {str(e)}")
                
        return documents
        
    def initialize_vector_store(self):
        """Initialize the vector store with documents."""
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            print("No documents found to index")
            return
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks")
        
        # Create vector store
        self.vector_store = FAISS.from_documents(
            chunks,
            self.embeddings
        )
        print("Vector store initialized successfully")
        
    def search(self, query: str, k: int = 3) -> List[tuple[Document, float]]:
        """Search the vector store for relevant documents."""
        if not self.vector_store:
            print("Vector store not initialized")
            return []
            
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"Error during vector search: {str(e)}")
            return []
            
    def get_relevant_context(self, query: str, k: int = 1) -> tuple[str, float, str, int]:
        """Get relevant context from documents for a query."""
        results = self.search(query, k=k)
        if not results:
            return "", 0.0, "", 0
        
        # Get the most relevant result
        doc, distance = results[0]
        # Convert L2 distance to similarity score (0 to 1)
        # Using the formula: similarity = 1 / (1 + distance)
        # This maps distance=0 to similarity=1, and distance=inf to similarity=0
        similarity = 1 / (1 + distance)
        
        # Extract metadata
        source = doc.metadata.get('source', 'unknown')
        chunk_id = doc.metadata.get('chunk_id', 0)
        
        return doc.page_content, similarity, source, chunk_id

# Global instance
document_processor = None

def initialize_vector_store():
    """Initialize the global document processor."""
    global document_processor
    document_processor = DocumentProcessor()
    document_processor.initialize_vector_store()
    
def get_document_processor() -> Optional[DocumentProcessor]:
    """Get the global document processor instance."""
    return document_processor 