from kafka import KafkaConsumer
import json
import logging
import os
from typing import Dict, Any
from evaluator.jobs.online import evaluate_current_run
from langsmith import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'evaluation-requests',
            bootstrap_servers=['kafka:29092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            group_id='evaluation-consumer-group'
        )
        self.ls_client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))
        
    def process_message(self, message: Dict[str, Any]) -> None:
        """Process a single evaluation request"""
        try:
            run_id = message['run_id']
            question = message['question']
            answer = message['answer']
            retrieved_contexts = message['retrieved_contexts']
            
            logger.info(f"Processing evaluation for run {run_id}, question: {question}")
            
            evaluate_current_run(run_id, question, answer, retrieved_contexts)

        except Exception as e:
            logger.error(f"Error processing evaluation request: {e}")
            
    def run(self):
        """Start consuming messages"""
        logger.info("Starting evaluation consumer...")
        
        try:
            for message in self.consumer:
                logger.info(f"Received message of length: {len(message.value)}")
                self.process_message(message.value)
                
        except KeyboardInterrupt:
            logger.info("Shutting down consumer...")
        finally:
            self.consumer.close()

if __name__ == "__main__":
    consumer = EvaluationConsumer()
    consumer.run() 