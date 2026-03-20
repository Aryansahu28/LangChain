from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from embedding_and_store import retrievedAnswer
from langchain_core.messages import HumanMessage
import base64
import cv2
import time
import logging



class LLM:

    def __init__(self,queries:list,video_path,model="qwen-vl:8b"):
        self.model = model
        self.queries = queries
        self.video_path = video_path

    def llm_initialization(self,image_data_list:list):
        model = ChatOllama(
            model="qwen3-vl:4b",
            temperature=0
        )

        context_info = " | ".join([f"Frame {i}: {d['timestamp']}" for i, d in enumerate(image_data_list)])
        
        content = [
            {"type": "text", "text": f"Context: {context_info}/nQuestion: {' '.join(self.queries)}"}
        ]
        
        for item in image_data_list:
            base64_image = self.encode_image(item['path'])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })

        # 5. Invoke the model
        message = HumanMessage(content=content)

        
        response = model.invoke([message])
        return response.content

    def encode_image(self,image_path,target_width=786):
        """ Convert images into base64encode to process"""
        img = cv2.imread(image_path)
        if img is None: return ""
        
        # Calculate new dimensions
        height, width = img.shape[:2]
        new_height = int(target_width * (height / width))
        resized = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode directly from the array to base64
        _, buffer = cv2.imencode('.jpg', resized)
        return base64.b64encode(buffer).decode('utf-8')
        
    
    def final_result(self):
        results = retrievedAnswer(self.video_path, self.queries)
        image_data_list = []
        for path, meta in zip(results['uris'][0], results['metadatas'][0]):
            image_data_list.append({'path': path, 'timestamp': meta['timestamp']})

        return self.llm_initialization(image_data_list) 


start_time = time.perf_counter()

llm = LLM(queries=["Which living thing is in the video and name of it species?"],video_path="C:/Users/HP-PC/Downloads/videos/insects/butterflies_1280.mp4")


        
print(llm.final_result())

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")





