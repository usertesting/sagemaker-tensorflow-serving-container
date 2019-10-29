curl -X POST --data-binary @test/resources/inputs/huiying2.json \
     -H 'Content-Type: application/json' \
     -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=huiying2' \
     http://localhost:8080/invocations
