from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertConfig, AlbertModel, BertTokenizer
from model import PairModel

def load_model():
    """载入模型，返回载入后的模型组件

    Returns:
        [dict] -- [模型组件]
    """
    print("** loading model.. **")
    tokenizer = BertTokenizer.from_pretrained(
        '../albert-small/', cache_dir=None, do_lower_case=True)
    bert_config = AlbertConfig.from_pretrained('../albert-small/')
    model = PairModel(config=bert_config)
    device = torch.device('cpu')
    state = torch.load(Path('../albert-small/pytorch_model.pt'),
                       map_location=device)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()
    return {'model': model,
            'tokenizer': tokenizer}


def convert_one_line(text_a, text_b, tokenizer):
    inputs = tokenizer.encode_plus(text_a, text_b, return_tensors='pt')
    token_ids = inputs['input_ids'][0]
    token_types = inputs['token_type_ids'][0]
    token_masks = inputs['attention_mask'][0]
    return token_ids, token_types, token_masks


def convert_batch(queries, candidates, tokenizer):
    token_ids, token_types, token_masks = [], [], []
    for q, c in zip(queries, candidates):
        token_id, token_type, token_mask = convert_one_line(
            q, c, tokenizer)
        token_ids.append(token_id)
        token_types.append(token_type)
        token_masks.append(token_mask)
    token_ids = pad_sequence(token_ids, batch_first=True)
    token_types = pad_sequence(token_types, batch_first=True, padding_value=1)
    token_masks = pad_sequence(token_masks, batch_first=True)
    return token_ids, token_types, token_masks


def batch_predict(batch_data, model_ctx):
    """批量推理

    Arguments:
        batch_data {List} -- [由task data组成的列表]
        model_ctx {Dict} -- [由load_model返回的字典]

    Returns:
        [array-like] -- [每个样本一个结果]
    """
    tokenizer = model_ctx['tokenizer']
    model = model_ctx['model']
    queries, candidates = zip(*batch_data)
    token_ids, token_types, token_masks = convert_batch(queries, candidates, tokenizer)
    with torch.no_grad():
        v_pred = model(token_ids, token_types, token_masks)
        v_pred = torch.sigmoid(v_pred)
    return v_pred.cpu().numpy().flatten()
