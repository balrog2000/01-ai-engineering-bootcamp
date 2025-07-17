from kafka import KafkaProducer
import json
import logging
from typing import Dict, Any
from api.core.config import config
from datetime import datetime

logger = logging.getLogger(__name__)

class EvaluationProducer:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=[f'{config.KAFKA_HOST}:{config.KAFKA_PORT}'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topic = config.KAFKA_TOPIC
        
    def publish_evaluation_request(self, run_id: str, question: str, answer: str, retrieved_contexts: list[str]) -> None:
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
        self.producer.close()

# Global publisher instance
if config.KAFKA_ENABLED:
    evaluation_producer = EvaluationProducer() 
else:
    evaluation_producer = None