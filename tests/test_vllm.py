'''
Make sure to set PYTHONPATH to the directory containing this file
when starting the vLLM server using trl vllm-serve ...
'''
import unittest

from trl.extras.vllm_client import VLLMClient

class LogitsProcessorTest(unittest.TestCase):
    def setUp(self):
        self.prompts = ["Hello, AI!", "Tell me a joke"]
        self.client = VLLMClient(server_port=3001)
        self.client.check_server()
    
    def test_logits_processor_func(self):
        responses = self.client.generate(self.prompts, n=1, max_tokens=32, logits_processors=['testing_utils.mock_logits_processor_func'])
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses, [[50] * 32] * 2) # Mocked response
        self.client.close_communicator()
    
    def test_logits_processor_class(self):
        responses = self.client.generate(self.prompts, n=1, max_tokens=32, logits_processors=[{'qualname': 'testing_utils.MockLogitsProcessorClass', 'args': [30]}])
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses, [[30] * 32] * 2) # Mocked response
        self.client.close_communicator()
