# SARA
Screening Agent for Rheumatoid Arthritis

## Setup
python3 -m venv venv
source venv/bin/activate(Linux) or venv\Scripts\activate(Windows)
pip3 install -r requirements.txt

create a .env in the project root folder
put the openai and google api keys in .env folder


| Model/Framework| Agent without KB|  Agent with KB | Two agents with KB |
|----------------|-----------------|----------------|--------------------|
| OpenAI O1      | ✅              | ✅            | ❌                 |
| OpenAI O3-mini | ✅              | ✅            | ❌                 |
| Gemini 2.5     | ✅              | ✅            | ❌                 |
| Gemini 2.0     | ✅              | ✅            | ❌                 |
| DeepSeek       | ✅              | ❌            | ❌                 |
| Qwen           | ✅              | ❌            | ❌                 |

