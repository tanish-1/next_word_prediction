from flask import Flask, render_template, request
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM

app = Flask(__name__)

def load_model():
    try:
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
        return bert_tokenizer, bert_model
    except Exception as e:
        pass

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

def get_all_predictions(tokenizer, model, text_sentence, top_k=5, top_clean=5):
    input_ids, mask_idx = encode(tokenizer, text_sentence)
    with torch.no_grad():
        predict = model(input_ids)[0]
    predictions = decode(tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return predictions

def get_prediction_eos(tokenizer, model, input_text, top_k=5):
    try:
        input_text += ' <mask>'
        res = get_all_predictions(tokenizer, model, input_text, top_k=top_k, top_clean=top_k) 
        return res
    except Exception as error:
        pass

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        top_k = None
        while top_k is None:
            try:
                top_k = int(request.form.get('top_k', 5))
                if top_k < 1 or top_k > 10:
                    return render_template('index.html', error="Please enter a number between 1 and 10.")
            except ValueError:
                return render_template('index.html', error="Please enter a valid integer.")

        bert_tokenizer, bert_model = load_model()

        input_text = request.form.get('input_text', '')
        res = get_prediction_eos(bert_tokenizer, bert_model, input_text, top_k=top_k)

        return render_template('index.html', prediction=res, input_text=input_text, top_k=top_k)
    except Exception as e:
        return render_template('index.html', error="SOME PROBLEM OCCURRED")

if __name__ == "__main__":
    app.run(debug=True)
