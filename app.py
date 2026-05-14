# pip install fastapi uvicorn transformers sentencepiece jinja2
# pip install jinja2
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import re

# Initialize FastAPI app
app = FastAPI(title="Text Summarization System", description="Summarize dialogues with T5!", version="1.0")


# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Ensure the model is on the correct device
device = "cuda" if model.device.type == "cuda" else "cpu"
model = model.to(device)

# Mount templates
templates = Jinja2Templates(directory="templates")


# Input schema for requests
class DialogueInput(BaseModel):
    dialogue: str


# Clean text function
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text


# Summarization function
def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_text(dialogue)
    
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="longest", max_length=512)
    
    # Ensure input tensors are on the same device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Generate summary
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_beams=4,
        early_stopping=True
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary



# API endpoint for text summarization
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


# HTML UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})