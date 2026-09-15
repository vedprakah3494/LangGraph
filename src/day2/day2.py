import sys
import ollama

model="gemma3"
request=input("Please enter what you want to discuss")
response = ollama.chat(
    model=model,
    messages=[
        {"role": "system", "content": "Classify the request in one word and also explain in detail in 100 word"},
        {"role": "user", "content": request},
    ],
)

print("Category:", response["message"]["content"])