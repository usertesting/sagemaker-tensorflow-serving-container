from sagemaker.tensorflow.serving import Model

from boto3.session import Session
session_boto = Session(region_name="us-west-1")

import sagemaker
sagemaker_session = sagemaker.Session(boto_session=session_boto)


model = Model(model_data='s3://ut-machine-learning-data/sentiment/tensorflow-models/huiying.tar.gz',
        role='AmazonSageMaker-ExecutionRole-20190822T174847',
        sagemaker_session=sagemaker_session
        )

# endpoint_name
predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge')
# Query

request = { "inputs": [ [1, 0], [0, 0]]}
print(predictor.predict(request))

##
import boto3
client = boto3.client('sagemaker-runtime', region_name='us-west-1')
#custom_attributes = "c000b4f9-df62-4c85-a0bf-7c525f9104a4"
endpoint_name = "sagemaker-tensorflow-serving-2019-10-28-19-14-41-613"
content_type = "application/json"
accept = "*/*"
payload = '{"inputs": [[1,0], [1,1]]}'
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    #CustomAttributes=custom_attributes,
    ContentType=content_type,
    Accept=accept,
    Body=payload
    )
