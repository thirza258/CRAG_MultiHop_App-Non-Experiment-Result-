import json
import asyncio
import logging
import threading

from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import redis

from router.models import Conversation, ConversationHistory, GuestUser
from rag.rag_service import get_registry
from common.pipeline_config import describe as describe_config
from common.pipeline_config import normalize_pipeline_config
from ragreader.settings import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

# Maximum time (seconds) the consumer will wait for the pipeline thread.
_PIPELINE_TIMEOUT = 300
# How long to wait between Redis poll attempts.
_POLL_INTERVAL = 0.5


class QueryStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, code):
        pass

    async def receive(self, text_data):
        pubsub = None
        r = None
        try:
            data     = json.loads(text_data)
            logger.info(f"Received WebSocket message: {data}")
            username = data.get("USER")
            query    = data.get("QUERY")
            # Normalised here rather than deeper in, so a malformed CONFIG from any
            # client degrades to the default pipeline instead of failing the query.
            pipeline_config = normalize_pipeline_config(data.get("CONFIG"))

            if not username or not query:
                await self.send(json.dumps({
                    "stage": "error", "message": "USER and QUERY required"
                }))
                return

            logger.info(
                f"Starting RAG pipeline for user: {username}, query: {query}, "
                f"config: {describe_config(pipeline_config)}"
            )

            conversation = await sync_to_async(self.save_conversation)(username, query, "", "", None)  
            task_id     = conversation.pk 
            channel = f"rag:status:{task_id}"

            import redis.asyncio as aioredis
            r      = aioredis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0,
                socket_connect_timeout=5, socket_timeout=5,
            )
            pubsub = r.pubsub()
            await pubsub.subscribe(channel)
            logger.info(f"Subscribed to Redis channel: rag:status:{task_id}")

            def run():
                # Created outside the try so the except block can always
                # publish the error back to the websocket loop.
                r_sync = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, db=0,
                    socket_connect_timeout=5, socket_timeout=10,
                )
                try:
                    answer = get_registry().get_engine().run(
                        username, query, conversation_id=task_id, config=pipeline_config
                    )
                    logger.info(f"RAG pipeline completed for task_id: {task_id}, answer: {answer}")

                    retrieved_chunks = answer.get("context", [])
                    llm_answer       = answer.get("answer", "")
                    context_str      = "\n\n".join(
                        doc["text"] if isinstance(doc, dict) else doc
                        for doc in retrieved_chunks
                    )
                    answer_record = self.save_conversation(
                        username, query, llm_answer, context_str, conversation_id=conversation.pk
                    )
                    
                    evaluation = answer.get("evaluation", {})

                    r_sync.publish(f"rag:status:{task_id}", json.dumps({
                        "stage": "result",
                        "answer": llm_answer,
                        "context": retrieved_chunks,
                        "conversation_id": answer_record.pk,
                        "evaluation": evaluation
                    }))

                except Exception as e:
                    logger.error(
                        f"Pipeline thread error for task {task_id}: {e}",
                        exc_info=True,
                    )
                    try:
                        r_sync.publish(f"rag:status:{task_id}", json.dumps({
                            "stage": "error",
                            "message": str(e)
                        }))
                    except Exception as pub_err:
                        logger.error(
                            f"Failed to publish error for task {task_id}: {pub_err}"
                        )
                finally:
                    try:
                        r_sync.close()
                    except Exception:
                        pass

            logger.info("Starting RAG pipeline thread")
            t = threading.Thread(target=run, daemon=True)
            t.start()
            logger.info("RAG pipeline thread started")
           
            elapsed = 0.0
            finished = False

            while elapsed < _PIPELINE_TIMEOUT:
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1.0
                    )
                except Exception as poll_err:
                    logger.warning(
                        f"Redis poll error (task {task_id}): {poll_err}"
                    )
                    await asyncio.sleep(_POLL_INTERVAL)
                    elapsed += _POLL_INTERVAL
                    continue

                if message is None:
                    await asyncio.sleep(_POLL_INTERVAL)
                    elapsed += _POLL_INTERVAL
                    continue

                logger.info(f"Got Redis message: {message}")

                if message["type"] != "message":
                    continue

                payload = json.loads(message["data"])
                await self.send(json.dumps(payload))
                logger.info(f"Sent to WebSocket: {payload.get('stage')}")

                if payload.get("stage") in ("result", "error"):
                    finished = True
                    break

            if not finished:
                # Don't leave the client hanging when the pipeline overruns.
                await self.send(json.dumps({
                    "stage": "error",
                    "message": f"Request timed out after {_PIPELINE_TIMEOUT}s. Please try again.",
                }))

        except Exception as e:
            logger.error(f"Consumer receive error: {e}", exc_info=True)
            try:
                await self.send(json.dumps({"stage": "error", "message": str(e)}))
            except Exception:
                pass  # WebSocket already closed
        finally:
            # Always clean up Redis resources
            if pubsub is not None:
                try:
                    await pubsub.unsubscribe()
                    await pubsub.aclose()
                except Exception:
                    pass
            if r is not None:
                try:
                    await r.aclose()
                except Exception:
                    pass
            logger.info("Done, connection closing cleanly")
            
    def save_conversation(self, username, query, answer, context, conversation_id=None):
        user = GuestUser.objects.get(username=username)

        if conversation_id:
            conversation = Conversation.objects.get(pk=conversation_id)
            conversation.response = answer
            conversation.context  = context
            conversation.save(update_fields=["response", "context"])
            return conversation
        else:
            conversation = Conversation.objects.create(
                user=user, 
                query=query, response="", context=""
            )
            ConversationHistory.objects.create(user=user, conversation=conversation)
            return conversation