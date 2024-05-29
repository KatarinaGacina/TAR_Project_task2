from transformers import BertTokenizerFast

def encode_data(data): #bert tokenization: tokens to numbers
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    inputs_ids = []
    
    for t in data:
        token_ids = tokenizer.convert_tokens_to_ids(list(t))
        inputs_ids.append(token_ids)

    return inputs_ids