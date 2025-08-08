# src/utils/api_utils.py

"""
Utility file for LoRA model API.
"""

from fastapi import FastAPI
from ollama import OllamaLLM
from transformers import GPTJForCausalLM, AutoTokenizer

app = FastAPI()

# Load the fine-tuned model
model = OllamaLLM(model="./models/fine_tuned_lora_model")

@app.post("/predict")
def predict(request: dict):
    user_input = request['text']  # Get the user input from the request
    response = model.generate(user_input)  # Use the model to generate a response
    return {"response": response}  # Return the generated response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)