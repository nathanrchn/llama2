# llama2
A python api to use the llama2-chat models hosted by the perplexity.ai team.

# Usage
```python
from sys import stdout
from llama import LLaMa

llama = LLaMa()

while True:
    p: str = str(input("> "))
    o = ""
    for packet in llama.chat(p, "13b"):
        stdout.write(packet["output"][len(o):])
        o = packet["output"]
        stdout.flush()
    stdout.write("\n\n")
```