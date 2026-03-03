!pip uninstall -y transformers
!pip install transformers==4.40.2 accelerate datasets

!pip uninstall -y transformers accelerate datasets huggingface_hub
!pip install --upgrade pip
!pip install transformers accelerate datasets huggingface_hub

!pip install gensim

# ============================================
# USER INPUT INTENT PREDICTION DEMO FOR mBERT
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_intent(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=90
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = torch.max(probabilities).item()

    intent_name = label_encoder.inverse_transform([predicted_class])[0]

    return intent_name, confidence


while True:
    user_input = input("\nEnter a sentence (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    intent, confidence = predict_intent(user_input)

    print("Predicted Intent:", intent)
    print("Confidence: {:.2f}%".format(confidence * 100))

from google.colab import drive
drive.mount('/content/drive')