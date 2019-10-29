curl -X POST --data-binary @test/resources/inputs/huiying.json \
     -H 'Content-Type: application/json' \
     -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=huiying' \
     http://localhost:8080/invocations
