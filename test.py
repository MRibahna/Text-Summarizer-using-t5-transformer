from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = "./saved_summary_model"

try:
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
