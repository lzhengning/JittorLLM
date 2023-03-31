import jittor as jt
import os
jt.flags.use_cuda = 1

from models import LLMModel
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

class ChatRWKVMdoel(LLMModel):
    def __init__(self, args) -> None:
        super().__init__()
        tokenizer_path = getattr(args, "tokenizer_path", os.path.join(os.path.dirname(os.path.realpath(__file__)), "20B_tokenizer.json"))
        model = RWKV(model=ckpt_dir, strategy='cpu fp32')
        self.generator = PIPELINE(model, tokenizer_path) # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV
        
        self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100,
                            alpha_frequency = 0.25,
                            alpha_presence = 0.25,
                            token_ban = [0], # ban the generation of some tokens
                            token_stop = []) # stop generation whenever you see any token here
        jt.gc()

    def run(self, input_text: str) -> str:
        def print_output(s):
            print(s, end='', flush=True)
        output = self.generator.generate(input_text, token_count=200, args=self.args, callback=print_output)
        return output

def get_model(args):
    args.ckpt_dir = args.paths
    return ChatRWKVMdoel(args)
