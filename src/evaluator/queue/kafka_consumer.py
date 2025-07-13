from kafka import KafkaConsumer
import json
import logging
import os
from typing import Dict, Any
from evaluator.jobs.online import evaluate_current_run, evaluate_current_run_native_ls
from langsmith import Client
from requests.exceptions import HTTPError
from evaluator.core.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            config.KAFKA_TOPIC,
            bootstrap_servers=[f'{config.KAFKA_HOST}:{config.KAFKA_PORT}'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None
        )
        self.ls_client = Client(api_key=config.LANGSMITH_API_KEY)
        
    def process_message(self, message: Dict[str, Any]) -> None:
        """Process a single evaluation request"""
        try:
            run_id = message['run_id']
            question = message['question']
            answer = message['answer']
            retrieved_contexts = message['retrieved_contexts']
            
            logger.info(f"Processing evaluation for run {run_id}, question: {question}")
            
            import time

            max_retries = 3
            retry_delay = 5  # seconds

            for attempt in range(max_retries):
                try:
                    # sometimes the run is not found, so we need to retry
                    evaluate_current_run_native_ls(run_id, question, answer, retrieved_contexts)
                    break  # Success, exit loop
                except Exception as e:
                    if "404" in str(e):
                        logger.warning(f"FORK2 404 error encountered during evaluation (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay} seconds...")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            logger.error("Max retries reached for 404 error. Skipping evaluation.")
                    else:
                        raise

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