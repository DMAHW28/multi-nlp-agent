import torch
import requests
import tensorflow as tf
from api.models import TextClassifierTransformer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForTokenClassification, AutoConfig, AutoTokenizer

MODEL_PATH_NER = '/ner_ncbi_lora_bert_model.pth'
MODEL_PATH_EMO = '/emotion_analyse_lora_bert_model.pth'
MODEL_PATH_SEN = '/sentiment_analyse_model_fs.pth'

MODEL_NAME_NER = "elastic/distilbert-base-uncased-finetuned-conll03-english"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME_NER)


# --------- Config ----------
DEVICE = torch.device('mps')
MODEL_DIR = "models"
EMO_DICO = {
    0: "triste",
    1: "joie",
    2: "amour",
    3: "colère",
    4: "peur",
    5: "surprise",
}
ID_TO_TAG = {
    0: 'O',
    1: 'B-Disease',
    2: 'I-Disease'
}

API_URL = "http://127.0.0.1:8000"
def call_api(end_point, payload):
    try:
        url = f"{API_URL}/{end_point}"
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return  response.json()["result"]
        else:
            return f"Erreur API : {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Impossible de se connecter à l'API."

def merge_tokens_and_labels(tokens, labels):
    """Fusionne les sous-tokens en mots et ajuste les labels."""
    words = []
    word_labels = []
    current_word = ""
    current_label = None

    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            current_word += token[2:]
        else:
            if current_word:
                words.append(current_word)
                word_labels.append(current_label)
            current_word = token
            current_label = ID_TO_TAG[label]

    if current_word:
        words.append(current_word)
        word_labels.append(current_label)

    clean_words = [TOKENIZER.convert_tokens_to_string([w]).strip() for w in words]
    return clean_words, word_labels

def ner_ncbi_pipeline(model, text, max_len=64):
    tokenized_inputs = TOKENIZER(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    mask = tokenized_inputs["attention_mask"]
    model.eval()
    with torch.inference_mode():
        outputs = model(**tokenized_inputs).logits
        y_pred = outputs.argmax(dim=-1).detach().cpu().tolist()

    if isinstance(text, str):
        m_len = mask.sum().item()
        pred_labels = y_pred[0][1:m_len-1]
        token_ids = tokenized_inputs["input_ids"][0, 1:m_len-1]
        tokens = TOKENIZER.convert_ids_to_tokens(token_ids)

        _, word_labels = merge_tokens_and_labels(tokens, pred_labels)
        pred_tag = ""
        for p in word_labels:
            pred_tag += f'{p} '

        return pred_tag
    else:
        raise ValueError("text must be str")

def emotion_pipeline(text, model, tokenizer, device=DEVICE, max_length=64):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    src_mask = torch.tensor(inputs["attention_mask"], dtype=torch.bool, device=device).unsqueeze(0)
    src_inp = torch.tensor(inputs["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(src_inp, src_mask)
        outputs_prob = torch.nn.functional.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(outputs_prob, dim=1).detach().cpu().tolist()
    return EMO_DICO[predictions[0]]

class Sentiment_pipeline:
    def __init__(self):
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        self.word_index["<UNUSED>"] = 3
        self.word_index = {word: index for word, index in self.word_index.items() if index < 5000}

    def text_to_sequence(self, text):
        words = text.lower().split()
        sequence = [self.word_index.get(word, 2) for word in words]
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=100, truncating='post')
        return padded_sequence

    def preprocessing(self, text):
        padded_sequence = self.text_to_sequence(text)
        inputs_ids = torch.tensor(padded_sequence, dtype=torch.long)
        mask = (inputs_ids != 0)
        mask = mask.to(dtype=torch.bool)
        return inputs_ids, mask

    def predict(self, input_ids, mask, model):
        with torch.inference_mode():
            model.eval()
            output = model(input_ids, mask)
        return output.argmax().detach().cpu().item()

    def pipeline(self, text, model):
        input_ids, mask = self.preprocessing(text)
        pred = self.predict(input_ids, mask, model)
        return "Positive" if pred == 1 else "Negative"


# --------- Load Models ----------
def load_models(device=DEVICE):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    output_dim = len(EMO_DICO)

    # BERT LORA
    emotion_bert_lora = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=output_dim)
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="SEQ_CLS")
    emotion_bert_lora = get_peft_model(emotion_bert_lora, lora_config)
    emotion_bert_lora.load_state_dict(torch.load(f"{MODEL_DIR + MODEL_PATH_EMO}", map_location=device))
    emotion_bert_lora.eval().to(device)


    config = AutoConfig.from_pretrained(MODEL_NAME_NER)
    config.num_labels = 3
    ner_ncbi_bert_lora = AutoModelForTokenClassification.from_pretrained(MODEL_NAME_NER, config=config, ignore_mismatched_sizes=True)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"],
        task_type=TaskType.TOKEN_CLS
    )
    ner_ncbi_bert_lora = get_peft_model(ner_ncbi_bert_lora, lora_config)
    ner_ncbi_bert_lora.load_state_dict(torch.load(f"{MODEL_DIR + MODEL_PATH_NER}", map_location=device))


    d_out: int = 2
    n_head: int = 2
    d_model: int = 64
    dropout: float = 0.75
    num_embeddings: int = 5000
    num_encoder_layers: int = 1
    dim_feedforward: int = 1 * d_model

    sentiment_analysis_model = TextClassifierTransformer(
        vocab_size=num_embeddings,
        dim_feedforward=dim_feedforward,
        output_dim=d_out,
        num_layers=num_encoder_layers,
        n_head=n_head,
        d_model=d_model,
        dropout=dropout
    )

    sentiment_analysis_model.load_state_dict(torch.load(f"{MODEL_DIR + MODEL_PATH_SEN}", map_location=device))

    models = {
        "Emotion": emotion_bert_lora,
        "Ner": ner_ncbi_bert_lora,
        "Sentiment": sentiment_analysis_model,
    }

    return models, tokenizer

