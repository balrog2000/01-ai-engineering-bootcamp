from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())

        request.state.request_id = request_id
        logger.info(f"Request started with ID: {request_id}")

        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info(f"Request finished with ID: {request.state.request_id}")
        
        return response