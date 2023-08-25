from requests import post
from json import dumps, loads
from random import getrandbits
from websocket import create_connection

from llama import LLaMa

class LLaMaHF(LLaMa):
    def __init__(self) -> None:
        self.history: list = []
        self.urls: dict = {
            "7b": "huggingface-projects-llama-2-7b-chat--glp8g.hf.space",
            "13b": "huggingface-projects-llama-2-13b-chat--sg2t4.hf.space"
        }
        self.session_hash: str = self.get_session_hash()
        self.default_system_prompt: str = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

    def get_session_hash(self) -> str:
        return format(getrandbits(44), "08x")
    
    def chat(self, prompt: str, model: str = "7b", max_new_tokens: int = 1024, temperature: float = 1, top_p: float = 0.95, top_k: int = 50, system_prompt: str = None) -> str:
        assert model in ["7b", "13b"], "Invalid model"

        if system_prompt is None:
            system_prompt = self.default_system_prompt

        post(
            url=f"https://{self.urls[model]}/run/predict",
            data=dumps({
                "data": [prompt],
                "event_data": None,
                "fn_index": 2,
                "session_hash": self.session_hash
            })
        )
        
        self.history.append([prompt, ""])
        
        ws = create_connection(f"wss://{self.urls[model]}/queue/join")
        
        send_hash_msg = ws.recv()
        assert send_hash_msg == '{"msg":"send_hash"}', "Failed to connect"

        ws.send(dumps({"fn_index": 5, "session_hash": self.session_hash}))

        # wait until receive the send_data_msg
        while True:
            msg = ws.recv()
            if msg == '{"msg":"send_data"}':
                break

        ws.send(dumps({
            "data": [
                None,
                self.history,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k
            ],
            "event_data": None,
            "fn_index": 5,
            "session_hash": self.session_hash
        }))

        output_length: int = 0
        while True:
            msg = loads(ws.recv())
            if msg["msg"] == "process_completed":
                self.history[-1][1] = msg["output"]["data"][0][-1][1]
                break
            elif msg["msg"] == "process_generating":
                output = msg["output"]["data"][0][-1][1]
                ooutput_length = output_length
                output_length = len(output)
                yield output[ooutput_length:]

        ws.close()