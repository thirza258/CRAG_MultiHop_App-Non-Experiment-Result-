import json
import asyncio
import logging
import threading

from channels.generic.websocket import AsyncWebsocketConsumer
from django.core.cache import cache
from asgiref.sync import sync_to_async
from django.core.exceptions import ObjectDoesNotExist
import redis

from router.models import AnalysisBatch, AnalysisResult, Conversation, ConversationHistory, GuestUser
from rag.rag_service import get_registry, rag_registry
from common.constant import CONFIG_VARIANTS
from ragreader.settings import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

class AnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        try:
            self.job_id = self.scope['url_route']['kwargs']['job_id']
            self.group_name = f"analysis_{self.job_id}"

            await self.channel_layer.group_add(
                self.group_name,
                self.channel_name
            )
            await self.accept()

            asyncio.create_task(self.run_rag_pipeline())
        except Exception as e:
            print(f"Error during WebSocket connection: {e}")
            await self.close()

    async def disconnect(self, close_code):
        try:
            await self.channel_layer.group_discard(
                self.group_name,
                self.channel_name
            )
        except Exception as e:
            print(f"Error during disconnect: {e}")

    async def run_rag_pipeline(self):
        try:
            input_data = await sync_to_async(cache.get)(f"job_input_{self.job_id}")
            
            if not input_data:
                await self.send(text_data=json.dumps({"error": "Job cache expired or invalid"}))
                await self.close()
                return

            username = input_data['username']
            query = input_data['query']
            document_id = input_data.get('document_id')
            conversation_id = input_data.get('conversation_id')

            try:
                analysis_batch = await sync_to_async(AnalysisBatch.objects.get)(job_id=self.job_id)
            except ObjectDoesNotExist:
                await self.send(text_data=json.dumps({"error": "Batch record not found in DB"}))
                await self.close()
                return

            existing_results = await sync_to_async(
                lambda: list(AnalysisResult.objects.filter(batch=analysis_batch))
            )()

            completed_variants = {
                (r.method, r.ai_model) for r in existing_results
            }

            if len(completed_variants) >= len(CONFIG_VARIANTS):
                await self.send(text_data=json.dumps({"status": "REPLAYING"}))
                for result in existing_results:
                    await self.send(text_data=json.dumps({
                        "batch_id": str(self.job_id),
                        "query": result.query,
                        "method": result.method,
                        "aiModel": result.ai_model,
                        "answer": result.answer,
                        "context": result.retrieved_chunks or [],
                        "progress": 100,
                        "replayed": True
                    }))
                await self.send(text_data=json.dumps({"status": "COMPLETE", "progress": 100}))
                await self.close()
                return

            total_variants = len(CONFIG_VARIANTS)

            for index, config in enumerate(CONFIG_VARIANTS):
                method = config["method"]
                model = config["model"]
                try:
                    if (method, model) in completed_variants:
                        existing = next(
                            r for r in existing_results
                            if r.method == method and r.ai_model == model
                        )
                        progress = int(((index + 1) / total_variants) * 100)
                        await self.send(text_data=json.dumps({
                            "batch_id": str(self.job_id),
                            "query": existing.query,
                            "method": existing.method,
                            "aiModel": existing.ai_model,
                            "answer": existing.answer,
                            "context": existing.retrieved_chunks or [],
                            "progress": progress,
                            "replayed": True
                        }))
                        continue

                    engine = rag_registry.get_engine(method, model)
                    
                    is_initialized = await sync_to_async(engine.is_initialized)(username)
                    
                    if not is_initialized:
                        await self.send(text_data=json.dumps({
                            "status": "INITIALIZING",
                            "method": method,
                            "aiModel": model,
                            "progress": int(((index + 0.5) / total_variants) * 100)
                        }))
                        await sync_to_async(engine.init)(username)

                    response = await sync_to_async(engine.run_analysis)(document_id, conversation_id)

                    llm_answer = response.get("answer", "")
                    context = response.get("context", [])
                    evaluation = response.get("evaluation", {})
                    
                    retrieved_chunks = [
                        {"id": doc["chunk_id"], "text": doc["text"], "score": doc.get("score")}
                        for doc in context
                    ]
                    
                    evaluation_with_retrieval = {
                        "chunk_evaluation": evaluation.get("chunk_evaluation", {}),
                        "response_evaluation": evaluation.get("response_evaluation", {}),
                        "retrieval_score": [
                            {"chunk_id": doc["chunk_id"], "score": doc.get("score")}
                            for doc in context
                        ]
                    }

                    metrics = [
                        {"name": key, "value": value}
                        for key, value in evaluation_with_retrieval.items()
                    ]

                    await sync_to_async(AnalysisResult.objects.create)(
                        batch=analysis_batch,
                        method=method,
                        ai_model=model,
                        answer=llm_answer,
                        query=query,
                        retrieved_chunks=retrieved_chunks,
                        evaluation_metrics=metrics
                    )

                    progress = int(((index + 1) / total_variants) * 100)
                    await self.send(text_data=json.dumps({
                        "batch_id": str(self.job_id),
                        "query": query,
                        "method": method,
                        "aiModel": model,
                        "answer": llm_answer,
                        "context": context,
                        "evaluation": evaluation_with_retrieval,
                        "progress": progress
                    }))

                except Exception as e:
                    await self.send(text_data=json.dumps({
                        "method": method,
                        "error": str(e),
                        "progress": int(((index + 1) / total_variants) * 100)
                    }))

            await self.send(text_data=json.dumps({"status": "COMPLETE", "progress": 100}))
            await self.close()
        except Exception as e:
            await self.send(text_data=json.dumps({"error": f"Pipeline error: {str(e)}"}))
            await self.close()

class QueryStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, code):
        pass

    async def receive(self, text_data):
        try:
            data     = json.loads(text_data)
            logger.info(f"Received WebSocket message: {data}")
            username = data.get("USER")
            query    = data.get("QUERY")

            if not username or not query:
                await self.send(json.dumps({
                    "stage": "error", "message": "USER and QUERY required"
                }))
                return
            
            logger.info(f"Starting RAG pipeline for user: {username}, query: {query}")

            conversation = await sync_to_async(self.save_conversation)(username, query, "", "", None)  
            task_id     = conversation.pk 
            channel = f"rag:status:{task_id}"

            import redis.asyncio as aioredis
            r      = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
            pubsub = r.pubsub()
            await pubsub.subscribe(channel)
            logger.info(f"Subscribed to Redis channel: rag:status:{task_id}")

            def run():
                # Created outside the try so the except block can always
                # publish the error back to the websocket loop.
                r_sync = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
                try:
                    answer = get_registry().get_engine().run(username, query, conversation_id=task_id)
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
                    r_sync.publish(f"rag:status:{task_id}", json.dumps({
                        "stage": "error",
                        "message": str(e)
                    }))
            logger.info("Starting RAG pipeline thread")
            threading.Thread(target=run).start()
            logger.info("RAG pipeline thread started")
           
            timeout = 300  # 5 minutes max
            elapsed = 0
            finished = False

            while elapsed < timeout:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)

                if message is None:
                    await asyncio.sleep(0.1)
                    elapsed += 0.1
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
                    "message": f"Request timed out after {timeout}s. Please try again.",
                }))

            await pubsub.unsubscribe(channel)
            await r.aclose()
            logger.info("Done, connection closing cleanly")

        except Exception as e:
            logger.error(f"Consumer receive error: {e}", exc_info=True)
            await self.send(json.dumps({"stage": "error", "message": str(e)}))
            
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