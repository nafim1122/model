# C-Code LLM Studio - Web UI

A modern, ChatGPT-style web interface for the C-Code LLM model.

## Features

- 🎨 **Modern UI** - Clean, dark/light mode ChatGPT-style interface
- ⚡ **Streaming Responses** - Token-by-token streaming for real-time feedback
- 🖥️ **Syntax Highlighting** - Prism.js powered C code highlighting
- 💾 **Conversation History** - localStorage persistence
- 📱 **Responsive Design** - Works on desktop and mobile
- 🎛️ **Multiple Modes** - Default, Strict C99, Educational, Optimized

## Quick Start

```bash
cd web-ui

# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Project Structure

```
web-ui/
├── src/
│   ├── app/
│   │   ├── api/generate/     # API route for model inference
│   │   ├── globals.css       # Global styles + Prism theme
│   │   ├── layout.tsx        # Root layout
│   │   └── page.tsx          # Main page
│   ├── components/
│   │   ├── ChatArea.tsx      # Chat messages + input
│   │   ├── Sidebar.tsx       # Conversations + settings
│   │   └── ThemeProvider.tsx # Dark/light mode
│   └── types/
│       └── index.ts          # TypeScript interfaces
├── package.json
├── tailwind.config.js
└── tsconfig.json
```

## Configuration

Edit `.env.local`:

```bash
# Use mock responses (demo mode)
USE_MOCK=true

# Connect to actual model
USE_MOCK=false
MODEL_ENDPOINT=http://localhost:8000/generate
```

## Connecting to Your Model

The API expects POST requests to `/api/generate` with:

```json
{
  "prompt": "Write a function to reverse a string",
  "systemPrompt": "You are a C code expert...",
  "model": "c-code-llm"
}
```

Response should be a stream of text (the generated C code).

### Python Model Server Example

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
model = AutoModelForCausalLM.from_pretrained("./runs/sft_lora")
tokenizer = AutoTokenizer.from_pretrained("./runs/sft_lora")

@app.post("/generate")
async def generate(request: dict):
    prompt = request["prompt"]
    
    async def stream():
        inputs = tokenizer(prompt, return_tensors="pt")
        for token in model.generate(**inputs, max_new_tokens=512, do_sample=True):
            yield tokenizer.decode(token)
    
    return StreamingResponse(stream(), media_type="text/plain")
```

## Tech Stack

- **Next.js 14** - App Router
- **React 18** - UI Framework
- **Tailwind CSS** - Styling
- **Prism.js** - Code highlighting
- **Lucide Icons** - Icons
- **TypeScript** - Type safety

## Screenshots

### Dark Mode
![Dark Mode](./screenshots/dark.png)

### Light Mode  
![Light Mode](./screenshots/light.png)

## License

MIT
