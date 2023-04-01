import os
from transformers import AutoTokenizer, AutoModel
from models import LLMModel

class ChatGLMMdoel(LLMModel):
    def __init__(self, args) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(__file__), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(os.path.dirname(__file__), trust_remote_code=True).half().cuda()
        self.model = self.model.eval()

    def run(self, text, history=[]):
        return self.model.chat(self.tokenizer, text, history=history)
    
    def chat(self) -> str:
        history = []
        while True:
            text = input("用户输入:")
            response, history = self.run(text, history=history)
            print("Chat-GLM: ", end='')
            print(response)
            print("")
    
    def run_web_demo(self, input_text, history=[]):
        while True:
            yield self.run(input_text, history=history)

def get_model(args):
    return ChatGLMMdoel(args)