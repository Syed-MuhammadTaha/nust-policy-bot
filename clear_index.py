"""
Utility script to clear the Qdrant hybrid search collection and reset the system.
Use this when you want to reprocess documents or start fresh.
"""

import os
from qdrant_client import QdrantClient
from config import Config

def clear_qdrant_collection():
    """Delete the Qdrant hybrid search collection and processed files metadata."""
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            host=Config.QDRANT_HOST,
            port=Config.QDRANT_PORT
        )
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if Config.QDRANT_COLLECTION_NAME in collection_names:
            # Get collection info before deleting
            collection_info = client.get_collection(Config.QDRANT_COLLECTION_NAME)
            points_count = collection_info.points_count
            
            # Delete the collection
            client.delete_collection(Config.QDRANT_COLLECTION_NAME)
            print(f"✅ Deleted Qdrant collection: {Config.QDRANT_COLLECTION_NAME}")
            print(f"   - Removed {points_count} points (chunks)")
            print(f"   - Embedding types: Dense, Sparse (BM25), Late Interaction (ColBERT)")
        else:
            print(f"ℹ️  Collection '{Config.QDRANT_COLLECTION_NAME}' does not exist")
        
        # Delete processed files metadata
        if os.path.exists(Config.PROCESSED_FILES_PATH):
            os.remove(Config.PROCESSED_FILES_PATH)
            print(f"✅ Deleted metadata file: {Config.PROCESSED_FILES_PATH}")
        
        # Remove metadata directory if empty
        metadata_dir = os.path.dirname(Config.PROCESSED_FILES_PATH)
        if metadata_dir and os.path.exists(metadata_dir):
            try:
                if not os.listdir(metadata_dir):
                    os.rmdir(metadata_dir)
                    print(f"✅ Removed empty directory: {metadata_dir}")
            except:
                pass
        
        print("\n🎉 Successfully cleared all vector store data!")
        print("📝 Next time you run the app, all documents will be reprocessed")
        print("   with hybrid embeddings (Dense + Sparse + Late Interaction).")
        
    except Exception as e:
        print(f"❌ Error clearing Qdrant: {e}")
        print("\n💡 Make sure Qdrant is running:")
        print("   docker-compose up -d")


if __name__ == "__main__":
    print("🗑️  Clearing Qdrant Hybrid Search Collection...")
    print(f"   Collection: {Config.QDRANT_COLLECTION_NAME}")
    print(f"   Host: {Config.QDRANT_HOST}:{Config.QDRANT_PORT}")
    print(f"   Embedding Types: Dense, Sparse (BM25), Late Interaction (ColBERT)")
    print()
    
    response = input("Are you sure you want to delete all indexed documents? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        clear_qdrant_collection()
    else:
        print("❌ Cancelled. No changes made.")
