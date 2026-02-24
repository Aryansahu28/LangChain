from langchain_text_splitters import CharacterTextSplitter

text="""
Vector search is a common way to store and search over unstructured data (such as unstructured text). The idea is to store numeric vectors that are associated with the text. Given a query, we can embed it as a vector of the same dimension and use vector similarity metrics (such as cosine similarity) to identify related text.
LangChain supports embeddings from dozens of providers. These models specify how text should be converted into a numeric vector. Let’s select a model
"""

splitter = CharacterTextSplitter(
    chunk_size=20,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)

print(result)