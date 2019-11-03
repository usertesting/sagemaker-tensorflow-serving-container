import os
import json
import logging
import requests
from tokenization import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

vocab_fp = os.path.join(os.path.dirname(__file__), "vocab.txt")
max_seq_len = 128
tokenizer = FullTokenizer(vocab_fp)
log.info('Vocabulary loaded from : {}'.format(vocab_fp))

def handler(data, context):
    """Handle request.
        Args:
            data (obj): the request data
            context (Context): an object containing request and configuration details
        Returns:
            (bytes, string): data to return to client, (optional) response content type
        """
    processed_input = _input_handler(data, context)
    response = requests.post(context.rest_uri, data=processed_input)
    return _output_handler(response, context)

def _input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        body = json.loads(data.read().decode('utf-8'))
        log.debug("Received a request with {} sentences".format(len(body["inputs"])))
        log.debug("Received request body: {}".format(body))
        batch_ids, batch_mask, batch_seg = transform(body["inputs"])
        tfinput = {
                   "inputs": {
                       "input_ids": batch_ids,
                       "input_mask": batch_mask,
                       "input_type_ids": batch_seg
                   }
                  }
        log.debug("Plain text inputs transformed into: {}".format(tfinput))

        return json.dumps(tfinput)

def transform(sentences):
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

def present(text):
    outputs = json.loads(text)["outputs"]
    pretty_outputs = [ { "positive": output[0], "negative": output[1], "neutral": output[2]  } for output in outputs ]
    return json.dumps(pretty_outputs)

def _output_handler(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = context.accept_header
    prediction = data.content
    return present(prediction), response_content_type



if __name__ == "__main__":
    sentences = [ "This is very good", "This is not very clear", "This is confusing" ]
    output = transform(sentences)
    print(output)
