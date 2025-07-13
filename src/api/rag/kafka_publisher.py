from kafka import KafkaProducer
import json
import logging
from typing import Dict, Any
from api.core.config import config
from datetime import datetime

logger = logging.getLogger(__name__)

class EvaluationPublisher:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['kafka:29092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topic = "evaluation-requests"
        
    def publish_evaluation_request(self, run_id: str, question: str, answer: str, retrieved_contexts: list[str]) -> None:
        """Publish evaluation request to Kafka topic"""
        message = {
            "run_id": run_id,
            "question": question,
            "answer": answer,
            "retrieved_contexts": retrieved_contexts,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.producer.send(
                topic=self.topic,
                key=run_id,
                value=message
            )
            logger.info("Evaluation request published to Kafka.")
        except Exception as e:
            logger.error(f"Failed to publish evaluation request to Kafka: {e}")
            
    def close(self):
        """Close the Kafka producer"""
        self.producer.close()

# Global publisher instance
evaluation_publisher = EvaluationPublisher() 