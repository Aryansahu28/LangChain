from langchain_community.document_loaders.youtube import TranscriptFormat,YoutubeLoader


loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=HOXq7mLjhBU", 
    add_video_info=False,
    language=["en", "id"],

)
docs = loader.load()

print(docs[0])