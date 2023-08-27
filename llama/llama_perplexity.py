from time import sleep
from threading import Thread
from json import loads, dumps
from collections import deque
from random import getrandbits
from websocket import WebSocketApp
from requests import Session, Response

from .llama import LLaMa

class LLaMaPerplexity(LLaMa):
    def __init__(self) -> None:
        self.history: list = []
        self.finished: bool = True
        self.session: Session = Session()
        self.headers: dict = { "User-Agent": "" }
        self.t: str = format(getrandbits(32), "08x")

        self.queue: deque = deque()
        self.sid: str = self.get_sid()
        assert self.ask_anonymous_user(), "Failed to ask anonymous user"

        self.websocket: WebSocketApp = None
        self.create_websocket()
        sleep(0.1)

    def get_sid(self) -> str:
        response: Response = self.session.get(
            url="https://labs-api.perplexity.ai/socket.io/?transport=polling&EIO=4",
            headers=self.headers)

        assert response.status_code == 200, "Failed to get sid"
        return loads(response.text[1:])["sid"]
    
    def ask_anonymous_user(self) -> bool:
        response = self.session.post(
            url=f"https://labs-api.perplexity.ai/socket.io/?EIO=4&transport=polling&t={self.t}&sid={self.sid}",
            data="40{\"jwt\":\"anonymous-ask-user\"}",
            headers=self.headers
        ).text

        return response == "OK"
    
    def get_cookies_str(self) -> str:
        cookies = ""
        for key, value in self.session.cookies.get_dict().items():
            cookies += f"{key}={value}; "
        return cookies[:-2]
    
    def create_websocket(self) -> None:
        def on_message(ws, message):
            if message == "2":
                ws.send("3")
            elif message.startswith("42"):
                message = loads(message[2:])[1]
                if "status" not in message:
                    self.queue.append(message)
                elif message["status"] == "completed":
                    self.finished = True
                    self.history.append({"role": "assistant", "content": message["output"], "priority": 0})

        def on_open(ws):
            ws.send("2probe")
            ws.send("5")

        self.websocket = WebSocketApp(
            url=f"wss://labs-api.perplexity.ai/socket.io/?EIO=4&transport=websocket&sid={self.sid}",
            header={"User-Agent": "", "Cookie": self.get_cookies_str()},
            on_message=on_message,
            on_open=on_open)

        Thread(target=self.websocket.run_forever).start()

    def chat(self, prompt: str, model: str = "7b") -> dict:
        assert model in ["7b", "13b", "70b"], "Invalid model"
        self.history.append({"role": "user", "content": prompt, "priority": 0})

        self.finished = False
        model_name = "llama-2-" + model + "-chat"

        self.websocket.send("42[\"perplexity_playground\",{\"model\":\"" + model_name + "\",\"messages\":" + dumps(self.history) + "}]")

        while not self.finished:
            if len(self.queue) > 0:
                yield self.queue.popleft()

    def code(self, prompt: str) -> dict:
        self.history.append({"role": "user", "content": prompt, "priority": 0})

        self.finished = False
        self.websocket.send("42[\"perplexity_playground\",{\"model\":\"codellama-34b-instruct\",\"messages\":" + dumps(self.history) + "}]")

        while not self.finished:
            if len(self.queue) > 0:
                yield self.queue.popleft()

    def close(self) -> None:
        self.websocket.close()