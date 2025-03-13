from llm_api import get_completion
from utils import numpy_to_data_url

class Agent():
    def __init__(self, id, env_prompt, recency=5):
        self.id = id
        self.memory = []
        self.recency = recency
        self.response = None
        self.context = None
        self.env_prompt = env_prompt
    
    def observe(self, observation, shared_message_pool):
        observation_url = numpy_to_data_url(observation)
        current_context = {
            "role": "user",
            "content": [
                {"type": "text", "text": shared_message_pool},
                {"type": "image_url", 'image_url': {'url': observation_url}}
                ]
            }
        self.memory.append(current_context)
        
    def think(self):
        self.context = self.env_prompt + self.memory[-self.recency:]
        self.response = get_completion(self.context)
        
    def say(self):
        return self.response.communication_message
        
    def act(self):
        return self.response.action.value
    
    def __str__(self):
        return (
            f"Agent {self.id}"
            f"\n\tRecent Memories: \n\t\t{self.memory[-self.recency:]}"
            f"\n\tContext: \n\t\t{self.context}"
            f"\n\tResponse: \n\t\t{self.response}"
        )