import json

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data, in format of dict or string
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/json':
        print("I am used")
        # pass through json (assumes it's correctly formed)
        d = data.read().decode('utf-8')

        return d #{ "input": to_input(d) }

    if context.request_content_type == 'text/csv':
        # very simple csv handler
        return json.dumps({
            'instances': [float(x) for x in data.read().decode('utf-8').split(',')]
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))

def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    import json

    response_content_type = context.accept_header
    prediction = data.content

    return to_output(prediction), response_content_type


def to_input(data):
    import json
    d = json.loads(data)
    return json.dumps(d["inputs"])

def test_input():
    from collections import namedtuple
    with open("test/resources/inputs/huiying.json", 'rb') as f:
        Context = namedtuple("Context", "request_content_type")
        print(input_handler(f, Context('application/json')))

def to_output(content):
    data_dict = json.loads(content.decode())
    outputs = data_dict["outputs"]

    def predict(v):
        return 1 if v > 0.5 else 0

    def per_row(row):
        return [ predict(col) for col in row ]

    data_dict["outputs"] = [ per_row(r) for r in outputs]
    return json.dumps(data_dict, indent=1).encode()

if __name__ == "__main__":
    pass

