import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from frame_extractor import extract_frames_with_metadata
import torch

# 1. Initialize the OpenCLIP Embedding Function
# By default, this downloads a highly efficient Vision Transformer (ViT)
# It will run locally on your GPU and embed the actual visual contents.

def retrievedAnswer(path,queries:list):
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version (torch): {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available - reason unknown, check below")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} for embeddings")
    print("Embedding start")
    embedding_function = OpenCLIPEmbeddingFunction(device=device)

    # 2. We need an ImageLoader so Chroma knows how to read the physical files
    print("Image Loader")
    image_loader = ImageLoader()

    # 3. Create the database connection
    client = chromadb.PersistentClient(path="./local_visual_db")

    print("Chroma collection")
    collection = client.get_or_create_collection(
        name="pure_visual_frames",
        embedding_function=embedding_function,
        data_loader=image_loader
    )

    extracted_metadata  = extract_frames_with_metadata(path)
    # 4. Add your frames (using the list we made in the previous step)
    ids = [f"frame_{i}" for i in range(len(extracted_metadata))]
    paths = [item['path'] for item in extracted_metadata]
    metadatas = [{"timestamp": item['timestamp']} for item in extracted_metadata]

    # Notice we use 'uris' instead of 'documents'. 
    print("Collection add")
    # Chroma will open the images, pass them through the local Vision model, and save the vectors!
    collection.add(
        ids=ids,
        uris=paths,
        metadatas=metadatas
    )

    result = collection.query(
        query_texts=queries,
        n_results=4,
        include=["uris","metadatas"]
    )

    print(f"{result}")
    return result




if __name__ == "__main__":
    retrievedAnswer("C:/Users/HP-PC/Downloads/8064146-uhd_1440_1800_30fps.mp4",["Which is the happening in this video?"])


