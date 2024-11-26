import os
import logging
import time
from typing import List, Dict, Any
import re

# Machine Learning and NLP Libraries
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import whisper
import pyaudio
import numpy as np
import torch
import pandas as pd
from pypdf import PdfReader

import speech_recognition as sr
import pyttsx3

class RobotRAGSystem:
    def __init__(
        self, 
        embedding_model: str = 'all-MiniLM-L6-v2',
        chunk_size: int = 300, 
        overlap: int = 50,
        gemini_api_key: str = None,
        robot_name: str = "RoboAssistant"
    ):
        """
        Initialize comprehensive robot RAG system
        
        Args:
            embedding_model (str): Sentence transformer model
            chunk_size (int): Text chunk size
            overlap (int): Chunk overlap
            gemini_api_key (str): Google AI API key
            robot_name (str): Name of the robot
        """
        # Logging setup
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # ChromaDB setup
        storage_path = os.path.join(os.getcwd(), "chromadb_storage")
        os.makedirs(storage_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=storage_path)
        
        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Configuration
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # ChromaDB Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Document tracking
        self.document_metadata = {}
        
        # Gemini Setup
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            self.logger.error("Gemini API key must be provided")
            raise ValueError("Gemini API key must be provided")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Replace Whisper with simple speech recognition
        self.speech_recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.robot_name = robot_name
        
        # Initialize and configure text-to-speech
        self.tts_engine = pyttsx3.init()
        
        # Get available voices
        voices = self.tts_engine.getProperty('voices')
        
        # Configure voice properties
        # Try to find a suitable voice (usually the second voice is female)
        for voice in voices:
            if "english" in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Adjust speech rate (lower number = slower speech)
        self.tts_engine.setProperty('rate', 150)  # Default is 200
        
        # Adjust volume
        self.tts_engine.setProperty('volume', 0.9)  # Range: 0.0 to 1.0
    
    def load_documents(self, document_paths: List[str]) -> None:
        """
        Load documents from multiple sources
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
                'answer': f"Hello! I'm {self.robot_name}. I'm here to help answer your questions about the loaded documents. What would you like to know?",
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
    
    def listen_for_speech(self) -> str:
        """
        Listen for speech input and convert to text
        """
        with sr.Microphone() as source:
            print("\nListening...")
            # Adjust for ambient noise
            self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                # Listen for audio input
                audio = self.speech_recognizer.listen(source, timeout=5, phrase_time_limit=10)
                # Convert speech to text using Google's speech recognition
                text = self.speech_recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.WaitTimeoutError:
                print("No speech detected")
                return ""
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return ""
    
    def run_interactive_loop(self):
        """
        Interactive loop with Indian English greeting
        """
        welcome_message = f"Namaste! I am {self.robot_name}. I am here to help you with your questions!"
        print(welcome_message)
        self.speak_response(welcome_message)
        
        while True:
            try:
                # Get input (voice or typed)
                query = self.listen_for_speech()
                
                # Check for exit commands
                if query.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                    farewell = "Thank you for your time. Have a good day!"
                    print("\n" + farewell)
                    self.speak_response(farewell)
                    break
                
                if query:
                    # Process the query
                    result = self.process_query(query)
                    
                    # Print answer
                    print("\n--- Answer ---")
                    print(f"Answer: {result['answer']}")
                    print(f"Sources: {result['sources']}")
                    
                    # Speak the answer with pauses
                    self.speak_response(result['answer'])
                    
            except KeyboardInterrupt:
                print("\nShutting down the system. Thank you for your time!")
                self.speak_response("Thank you for your time. Have a good day!")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                self.speak_response("I apologize, there seems to be an error. Please try again.")
                time.sleep(1)
    
    def speak_response(self, text: str):
        """
        Convert text to speech with Indian English style formatting
        """
        try:
            # Pre-process text to improve pronunciation
            text = self._format_for_indian_english(text)
            
            # Speak with pauses at punctuation
            sentences = text.split('.')
            for sentence in sentences:
                if sentence.strip():
                    self.tts_engine.say(sentence.strip())
                    self.tts_engine.runAndWait()
                    time.sleep(0.5)  # Add pause between sentences
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            print("Continuing with text-only output...")
    
    def _format_for_indian_english(self, text: str) -> str:
        """
        Format text to improve Indian English pronunciation
        """
        # Common Indian English pronunciation adjustments
        replacements = {
            'data': 'daata',
            'process': 'prosess',
            'project': 'projekt',
            'system': 'sistum',
            'hello': 'helo',
            'hi': 'hai',
            'thanks': 'thank you',
            'okay': 'ok',
        }
        
        # Apply replacements
        formatted_text = text
        for old, new in replacements.items():
            formatted_text = re.sub(r'\b' + old + r'\b', new, formatted_text, flags=re.IGNORECASE)
        
        return formatted_text

def main():
    # Load environment variables
    load_dotenv()
    
    # Create robot RAG system
    robot_rag = RobotRAGSystem(robot_name="RoboAssistant")
    
    # Load documents
    document_paths = [
        'data/1.pdf',
        'data/2.pdf'
    ]
    robot_rag.load_documents(document_paths)
    
    # Run interactive loop
    robot_rag.run_interactive_loop()

if __name__ == "__main__":
    main()