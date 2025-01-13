import chromadb
from typing import Dict, Any, List
from .base_agent import BaseAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
import pypdf
import docx2txt
from chromadb.api.models.Collection import Collection

class IndexAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        print("Initializing IndexAgent...")
        self.client = chromadb.PersistentClient(path="./data/chroma")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print("\n=== IndexAgent: Starting Processing ===")
            files = state.get('files', [])
            print(f"Processing {len(files)} files")
            
            if not files:
                print("No files to process, skipping indexing")
                return state

            collection_name = f"session_{abs(hash(str(files)))}"  # Use abs to avoid negative hash
            print(f"Creating collection: {collection_name}")
            
            try:
                # Process documents first
                documents = []
                metadatas = []
                for file_path in files:
                    print(f"\nProcessing file: {file_path}")
                    text = self._read_file(file_path)
                    if text:
                        print(f"File content length: {len(text)} characters")
                        chunks = self.text_splitter.split_text(text)
                        print(f"Split into {len(chunks)} chunks")
                        documents.extend(chunks)
                        # Add metadata for each chunk
                        metadatas.extend([{"source": str(file_path)} for _ in chunks])
                    else:
                        print("Warning: No content extracted from file")

                if documents:
                    print(f"\nGenerating embeddings for {len(documents)} documents...")
                    embeddings = self.embeddings.embed_documents(documents)
                    print("Embeddings generated successfully")
                    
                    # Delete collection if it exists
                    try:
                        self.client.delete_collection(collection_name)
                        print("Deleted existing collection")
                    except:
                        pass

                    print("Creating new collection...")
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    
                    print("Adding documents to collection...")
                    # Add in batches to avoid memory issues
                    batch_size = 100
                    for i in range(0, len(documents), batch_size):
                        try:
                            end_idx = min(i + batch_size, len(documents))
                            batch_num = i//batch_size + 1
                            print(f"Processing batch {batch_num}...")
                            
                            collection.add(
                                embeddings=embeddings[i:end_idx],
                                documents=documents[i:end_idx],
                                metadatas=metadatas[i:end_idx],
                                ids=[f"doc_{j}" for j in range(i, end_idx)]
                            )
                            print(f"Successfully added batch {batch_num}")
                            
                        except Exception as e:
                            print(f"Error adding batch {i//batch_size + 1}: {str(e)}")
                            # Continue with next batch rather than failing completely
                            continue

                    print("Documents added to collection")
                    
                    # Create a wrapper that mimics the as_retriever interface
                    collection_wrapper = ChromaCollectionWrapper(collection)
                    state['index_collection'] = collection_wrapper
                    print("Collection added to state")
                else:
                    print("Warning: No documents to process")
                
            except Exception as e:
                print(f"!!! Error in collection processing: {str(e)}")
                raise

            print("=== IndexAgent: Processing Complete ===\n")
            return state
            
        except Exception as e:
            print(f"!!! Error in IndexAgent: {str(e)}")
            raise

    def _read_file(self, file_path: str) -> str:
        """Read file content based on file type"""
        try:
            print(f"Reading file: {file_path}")
            file_path = Path(file_path)
            suffix = file_path.suffix.lower()
            
            if not file_path.exists():
                print(f"!!! File not found: {file_path}")
                return ""
            
            try:
                if suffix == '.pdf':
                    return self._read_pdf(file_path)
                elif suffix == '.docx':
                    return docx2txt.process(str(file_path))
                elif suffix in ['.txt', '.md']:
                    return file_path.read_text(encoding='utf-8')
                else:
                    print(f"!!! Unsupported file type: {suffix}")
                    return ""
            except Exception as e:
                print(f"!!! Error reading file content: {str(e)}")
                return ""
                
        except Exception as e:
            print(f"!!! Error in _read_file: {str(e)}")
            return ""

    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file content"""
        try:
            print(f"Reading PDF: {file_path}")
            text = []
            with open(file_path, 'rb') as file:
                pdf = pypdf.PdfReader(file)
                print(f"PDF has {len(pdf.pages)} pages")
                for page in pdf.pages:
                    text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            print(f"!!! Error reading PDF: {str(e)}")
            return ""

    def validate_input(self, state: Dict[str, Any]) -> bool:
        return 'files' in state 

class ChromaCollectionWrapper:
    """Wrapper class to provide retriever interface for ChromaDB collection"""
    def __init__(self, collection: Collection):
        self.collection = collection

    def as_retriever(self):
        """Return self as the retriever"""
        return self

    def get_relevant_documents(self, query: str) -> List[Dict]:
        """Get relevant documents for a query"""
        print(f"Retrieving documents for query: {query[:100]}...")
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=5,
                include=['documents', 'metadatas']
            )
            
            # Format results to match expected interface
            documents = []
            if results and results['documents']:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    documents.append({
                        'page_content': doc,
                        'metadata': metadata
                    })
            print(f"Retrieved {len(documents)} relevant documents")
            return documents
        except Exception as e:
            print(f"!!! Error in document retrieval: {str(e)}")
            return [] 