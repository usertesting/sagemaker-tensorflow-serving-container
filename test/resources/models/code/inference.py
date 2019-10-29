from tokenization import *
#from tensorflow.contrib.util import make_tensor_proto, make_ndarray
import os, json

vocab_fp = os.path.join(os.path.dirname(__file__), "vocab.txt")
max_seq_len = 128
tokenizer = FullTokenizer(vocab_fp)

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        body = json.loads(data.read().decode('utf-8'))
        batch_ids, batch_mask, batch_seg = preprocessing(body["inputs"])

        return {
                   "inputs": {
                       "input_ids": batch_ids,
                       "input_mask": batch_mask,
                       "input_type_ids": batch_seg
                   }
                }

def preprocessing(sentences):
    batch_ids = []
    batch_mask = []
    batch_seg = []
    for sent in sentences:
        tks = tokenizer.tokenize(sent)
        if len(tks) > max_seq_len - 2:
            tks = tks[:(max_seq_len - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tks:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len
        batch_ids.append(input_ids)
        batch_mask.append(input_mask)
        batch_seg.append(segment_ids)
    return batch_ids, batch_mask, batch_seg
    #pred = get_prediction(lang, batch_ids, batch_mask, batch_seg)
    #return pred


# def get_prediction(self, lang, ids, mask, typeids):
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = lang + "_sentiment"
#     request.inputs['input_ids'].CopyFrom(
#         make_tensor_proto(ids, shape=np.array(ids).shape))
#     request.inputs['input_mask'].CopyFrom(
#         make_tensor_proto(mask, shape=np.array(mask).shape))
#     request.inputs['input_type_ids'].CopyFrom(
#         make_tensor_proto(typeids, shape=np.array(typeids).shape))
#
#     res = self.stub.Predict(request, 100.0)
#     output = make_ndarray(res.outputs['output'])
#
#     return output

def output_handler(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type


if __name__ == "__main__":
    sentences = [ "This is very good", "This is not very clear", "This is confusing" ]
    output = preprocessing(sentences)
    print(output)
