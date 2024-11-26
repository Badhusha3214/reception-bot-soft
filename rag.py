import os
from dotenv import load_dotenv
import logging
from typing import List, Optional, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class EnhancedRAGSystem:
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 chunk_size: int = 300, 
                 overlap: int = 50,
                 gemini_api_key: str = None):
        """
        Initialize RAG system with ChromaDB, embedding model, and Gemini
        
        Args:
            embedding_model (str): Sentence transformer model for embeddings
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            gemini_api_key (str): Google API key for Gemini
        """
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Ensure storage directory exists
        storage_path = os.path.join(os.getcwd(), "chromadb_storage")
        os.makedirs(storage_path, exist_ok=True)
        
        # ChromaDB setup with updated configuration
        self.chroma_client = chromadb.PersistentClient(path=storage_path)
        
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Configuration
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Collection to store embeddings
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Document tracking
        self.document_metadata = {}
        
        # Initialize Gemini
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            raise ValueError("Gemini API key must be provided or set in GEMINI_API_KEY environment variable")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Configure safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def load_documents(self, document_paths: List[str]) -> None:
        """
        Load documents from multiple sources
        
        Args:
            document_paths (List[str]): List of file paths to load
        """
        for path in document_paths:
            if not os.path.exists(path):
                self.logger.warning(f"Document not found: {path}")
                continue
            
            if path.lower().endswith('.pdf'):
                self._load_pdf(path)
            elif path.lower().endswith('.txt'):
                self._load_txt(path)
            elif path.lower().endswith('.json'):
                self._load_json(path)
            else:
                self.logger.warning(f"Unsupported file type: {path}")
        
        self.logger.info(f"Loaded {len(self.document_metadata)} documents")
    
    def _load_pdf(self, pdf_path: str) -> None:
        """Extract text from PDF and chunk it"""
        try:
            reader = PdfReader(pdf_path)
            text = " ".join([page.extract_text() for page in reader.pages])
            chunks = self._chunk_text(text)
            
            # Store document metadata
            doc_id = os.path.basename(pdf_path)
            self.document_metadata[doc_id] = {
                'source': pdf_path,
                'type': 'pdf',
                'total_chunks': len(chunks)
            }
            
            # Add chunks to ChromaDB
            self._add_to_chroma(chunks, doc_id)
        except Exception as e:
            self.logger.error(f"Error loading PDF {pdf_path}: {e}")
    
    def _load_txt(self, txt_path: str) -> None:
        """Extract text from TXT and chunk it"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = self._chunk_text(text)
                
                # Store document metadata
                doc_id = os.path.basename(txt_path)
                self.document_metadata[doc_id] = {
                    'source': txt_path,
                    'type': 'txt',
                    'total_chunks': len(chunks)
                }
                
                # Add chunks to ChromaDB
                self._add_to_chroma(chunks, doc_id)
        except Exception as e:
            self.logger.error(f"Error loading TXT {txt_path}: {e}")
    
    def _load_json(self, json_path: str) -> None:
        """Extract text from JSON and chunk it"""
        try:
            df = pd.read_json(json_path)
            text = " ".join(df.apply(lambda row: ' '.join(map(str, row)), axis=1))
            chunks = self._chunk_text(text)
                
            # Store document metadata
            doc_id = os.path.basename(json_path)
            self.document_metadata[doc_id] = {
                'source': json_path,
                'type': 'json',
                'total_chunks': len(chunks)
            }
            
            # Add chunks to ChromaDB
            self._add_to_chroma(chunks, doc_id)
        except Exception as e:
            self.logger.error(f"Error loading JSON {json_path}: {e}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text (str): Input text to chunk
        
        Returns:
            List[str]: List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def _add_to_chroma(self, chunks: List[str], doc_id: str) -> None:
        """
        Add chunks to ChromaDB collection
        
        Args:
            chunks (List[str]): Text chunks
            doc_id (str): Document identifier
        """
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        # Add to ChromaDB collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{'source': doc_id} for _ in range(len(chunks))]
        )
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant chunks for a given query
        
        Args:
            query (str): Input query
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            List[str]: Most relevant text chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0]
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """
        Generate answer using Gemini and retrieved context
        
        Args:
            query (str): Input question
            context (List[str]): Retrieved context chunks
        
        Returns:
            str: Generated answer
        """
        prompt = f"""Based on the following context, please answer the question.
        
Context:
{' '.join(context)}

Question: {query}

Please provide a clear and concise answer based only on the information provided in the context."""

        try:
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating response with Gemini: {e}")
            return "Sorry, I encountered an error while generating the response."
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the RAG system
        
        Args:
            query (str): Input query
        
        Returns:
            Dict containing query results and metadata
        """
        # Handle greetings
        greetings = ['hi', 'hello', 'hey', 'greetings']
        if query.lower().strip() in greetings:
            return {
                'query': query,
                'answer': "Hello! I'm here to help answer your questions about the loaded documents. What would you like to know?",
                'context': [],
                'sources': []
            }
        
        # Regular RAG processing for non-greeting queries
        relevant_context = self.retrieve_relevant_context(query)
        answer = self.generate_answer(query, relevant_context)
        
        return {
            'query': query,
            'answer': answer,
            'context': relevant_context,
            'sources': list(set([
                self.document_metadata.get(chunk.split('_chunk_')[0], {}).get('source', 'Unknown')
                for chunk in relevant_context
            ]))
        }

def main():
    # Load the .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable")
        return

    # Initialize RAG system with Gemini
    rag_system = EnhancedRAGSystem(gemini_api_key=api_key)
    
    # Load documents
    document_paths = [
        'data/college_overview.pdf',
        'data/mission_statement.txt'
    ]
    
    print("Loading documents...")
    rag_system.load_documents(document_paths)
    print("Documents loaded! Ready to answer questions.")
    print("Enter your questions (press Ctrl+C to exit):")
    
    try:
        while True:
            query = input("\nYour question: ").strip()
            if query:
                result = rag_system.process_query(query)
                print("\n--- Answer ---")
                print(f"Answer: {result['answer']}")
                print(f"Sources: {result['sources']}")
    except KeyboardInterrupt:
        print("\nShutting down the RAG system. Goodbye!")

if __name__ == "__main__":
    main()