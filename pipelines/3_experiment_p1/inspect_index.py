import sys
from pathlib import Path
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent))

import config
from embeddings import MonolingualEmbedding

def inspect_collection(language):
    print(f"\n--- Inspecting {language} Collection ---")
    
    persist_dir = config.CHROMA_DB_DIR / f"{language.lower()}_index"
    if not persist_dir.exists():
        print(f"Index directory not found: {persist_dir}")
        return

    # Initialize Client directly to inspect without embedding function first
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=f"{language.lower()}_collection")
        count = collection.count()
        print(f"Total Documents: {count}")
        
        if count > 0:
            # Get first 5 documents
            results = collection.get(limit=5)
            ids = results['ids']
            documents = results['documents']
            metadatas = results['metadatas']
            
            for i in range(len(ids)):
                print(f"\n[Doc {i+1}] ID: {ids[i]}")
                print(f"Metadata: {metadatas[i]}")
                print(f"Content (First 200 chars):")
                print("-" * 40)
                print(documents[i][:200] + "..." if len(documents[i]) > 200 else documents[i])
                print("-" * 40)
        else:
            print("Collection is empty!")
            
    except Exception as e:
        print(f"Error inspecting collection: {e}")

if __name__ == "__main__":
    inspect_collection("Tamil")
    inspect_collection("Bengali")
