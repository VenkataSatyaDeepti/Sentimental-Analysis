from flask import Flask, request, render_template_string
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pyngrok import ngrok

app = Flask(__name__)

# Load the saved model and tokenizer
model_path = '/content/model'  # Update with your actual path
loaded_model = BertForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = BertTokenizer.from_pretrained(model_path)

def predict_sentiment(text):
    inputs = loaded_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return "Positive" if predicted_class == 1 else "Negative"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review_text = request.form['review']
        sentiment = predict_sentiment(review_text)
        return render_template_string('''
            <h1>Sentiment Analysis</h1>
            <form method="POST" action="/">
                <textarea name="review" placeholder="Enter a review..."></textarea><br>
                <button type="submit">Analyze Sentiment</button>
            </form>
            <h2>Review:</h2>
            <p>{{ review }}</p>
            <h2>Sentiment:</h2>
            <p>{{ sentiment }}</p>
        ''', review=review_text, sentiment=sentiment)
    return render_template_string('''
        <h1>Sentiment Analysis</h1>
        <form method="POST" action="/">
            <textarea name="review" placeholder="Enter a review..."></textarea><br>
            <button type="submit">Analyze Sentiment</button>
        </form>
    ''')


ngrok.set_auth_token('2kqC7WyPG8YqesaSklIKBODOvh1_6PaYUrMzv6UsK4ietUqk')  # Optional: if you have an ngrok account
public_url = ngrok.connect(5000)
print('Public URL:', public_url)
app.run(port=5000)
