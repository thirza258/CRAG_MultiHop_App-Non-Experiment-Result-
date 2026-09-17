"""Document lifecycle with real storage, DB, chunkers and an in-memory Chroma index.

Only the network/model boundaries and Celery dispatch are substituted. The task,
retrievers, stage composition and API response handling run as shipped.
"""
import io
import itertools
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import chromadb
from chromadb.config import Settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.utils import timezone
from datetime import timedelta
from PyPDF2 import PdfWriter
from PyPDF2.generic import DictionaryObject, NameObject, DecodedStreamObject

from common.runtime.context import RuntimeSettings, use_runtime
from common.runtime.test_handoff import FakeRedis
from pipeline.app_pipeline import AppRAGPipeline
from rag.config import load_pipeline_config
from router.models import Document, DocumentChunk, GuestUser, UserCollection
from router.tasks import build_index_task
from utils.insert_file import DataLoader


class Provider:
    def __init__(self):
        self.embedding_calls = []
        self.chat_calls = []
        self.embeddings = SimpleNamespace(create=self.embed)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_completion))

    def embed(self, **kwargs):
        self.embedding_calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(len(text)), 1.0, 0.5])
                                     for i, text in enumerate(kwargs['input'])])

    def chat_completion(self, **kwargs):
        self.chat_calls.append(kwargs)
        prompt = kwargs['messages'][0]['content']
        if 'SUFFICIENT' in prompt:
            answer = 'SUFFICIENT'
        else:
            context = prompt.split('Context:\n')[-1].split('\n\nQuestion:')[0]
            match = re.search(r'launch code is ([A-Z]+-\d+)', context)
            answer = f'The launch code is {match[1]}.' if match else 'The documents do not provide that answer.'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])


def pdf_bytes(text):
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    page = writer.pages[0]
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({
        NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f'BT /F1 12 Tf 30 200 Td ({text}) Tj ET'.encode())
    page[NameObject('/Contents')] = writer._add_object(stream)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


class DocumentFlowTests(TestCase):
    def setUp(self):
        media = tempfile.TemporaryDirectory()
        self.addCleanup(media.cleanup)
        settings = override_settings(MEDIA_ROOT=media.name)
        settings.enable()
        self.addCleanup(settings.disable)
        self.chroma = chromadb.EphemeralClient(Settings(allow_reset=True, anonymized_telemetry=False))
        self.chroma.reset()
        self.provider = Provider()
        self.user = GuestUser.objects.create(username='alice', email='alice@flow.test')
        self.other = GuestUser.objects.create(username='bob', email='bob@flow.test')
        self.patch('chroma.chroma_settings.get_client', return_value=self.chroma)
        self.patch('chroma.chroma_settings.chromadb.HttpClient', return_value=self.chroma)
        self.patch('ai_handler.openrouter.openrouter_api_key', return_value='test-key')
        self.patch('dense_rag.dense_rag.openrouter_client', return_value=self.provider)
        self.patch('ai_handler.llm.openrouter_client', return_value=self.provider)
        self.patch('common.runtime.handoff._client', return_value=FakeRedis())
        self.dispatch = self.patch('router.views.build_index_task.delay')
        self.patch('hybrid_rag.hybrid_rag.HybridRAG._load_reranker', return_value=None)
        self.pipeline = AppRAGPipeline(load_pipeline_config())
        registry = SimpleNamespace(get_engine=lambda: self.pipeline)
        self.patch('router.views.get_registry', return_value=registry)
        self.patch('router.tasks.get_registry', return_value=registry)
        self.pipeline.hybrid_rag._model = SimpleNamespace(
            rerank=lambda query, candidates, top_n: [{'index': i} for i in range(len(candidates))])
        for side in ('dense', 'sparse'):
            evaluator = mock.Mock()
            evaluator.evaluate.side_effect = lambda query, docs, metas: ('correct', docs, metas, 0.99)
            evaluator.score_docs.side_effect = lambda query, docs: [0.99] * len(docs)
            getattr(self.pipeline, f'{side}_corrective_rag').evaluator = evaluator
        self.pipeline._build_emitter = lambda _: SimpleNamespace(emit=lambda *args, **kwargs: None)

    def patch(self, target, **kwargs):
        patcher = mock.patch(target, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def upload(self, text='The launch code is ORCHID-72.', username='alice', config=None, file=None):
        payload = {'USER': username, 'TEXT': text, 'CONFIG': config or {}}
        if file is None:
            response = self.client.post('/api/v1/insert-text/', payload, content_type='application/json')
        else:
            response = self.client.post('/api/v1/insert-data/', {'USER': username, 'FILE': file})
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual(response.json()['data']['status'], 'pending')
        build_index_task(**self.dispatch.call_args.kwargs)
        document = Document.objects.get(pk=response.json()['data']['document_id'])
        self.assertEqual(document.status, 'ready')
        return document

    def query(self, **config):
        return self.client.post('/api/v1/query/', {
            'USER': 'alice', 'QUERY': 'What is the launch code?',
            'CONFIG': {'use_multi_hop': False, 'use_corrective': False,
                       'use_evaluation': False, 'use_reranker': False, **config},
        }, content_type='application/json')

    def test_pdf_upload_index_and_answer_use_the_uploaded_file(self):
        document = self.upload(file=SimpleUploadedFile('mission.pdf', pdf_bytes('The launch code is ORCHID-72.')))
        response = self.query(llm_model='custom/chat', temperature=0.4)
        self.assertEqual(response.status_code, 200)
        result = response.json()['data']
        self.assertIn('ORCHID-72', result['answer'])
        self.assertTrue(result['context'])
        self.assertTrue(all(c['metadata']['document_id'] == document.pk for c in result['context']))
        self.assertEqual(result['context'][0]['metadata']['title'], 'mission.pdf')
        self.assertEqual(self.provider.chat_calls[-1]['model'], 'custom/chat')
        self.assertEqual(self.provider.chat_calls[-1]['temperature'], 0.4)

    def test_missing_api_key_reaches_upload_status_without_retry_or_partial_index(self):
        from ai_handler.openrouter import MissingAPIKeyError

        response = self.client.post('/api/v1/insert-text/', {
            'USER': 'alice', 'TEXT': 'The launch code is ORCHID-72.',
        }, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        document_id = response.json()['data']['document_id']
        with mock.patch('dense_rag.dense_rag.openrouter_client', side_effect=MissingAPIKeyError('Add an OpenRouter API key.')), \
             mock.patch.object(build_index_task, 'retry') as retry:
            with self.assertRaises(MissingAPIKeyError):
                build_index_task(**self.dispatch.call_args.kwargs)
        retry.assert_not_called()
        status = self.client.get(f'/api/v1/document/{document_id}/alice/').json()['data']
        self.assertEqual(status['status'], 'failed')
        self.assertIn('API key', status['error_message'])
        self.assertFalse(DocumentChunk.objects.filter(document_id=document_id).exists())

    def test_all_stage_combinations_and_retrievers_answer_from_the_user_file(self):
        document = self.upload()
        self.upload('The launch code is OTHER-99.', username='bob')
        with mock.patch.object(AppRAGPipeline, 'evaluate', return_value=(0.9, 0.8)) as judge:
            for retriever, hop, corrective, rerank, evaluate in itertools.product(
                ('dense', 'sparse', 'both'), (False, True), (False, True), (False, True), (False, True)
            ):
                with self.subTest(retriever=retriever, hop=hop, corrective=corrective, rerank=rerank, evaluate=evaluate):
                    judge.reset_mock()
                    response = self.query(retrievers=retriever, use_multi_hop=hop,
                        use_corrective=corrective, use_reranker=rerank, use_evaluation=evaluate,
                        use_query_expansion=False, rerank_top_k=1)
                    result = response.json()['data']
                    self.assertIn('ORCHID-72', result['answer'])
                    self.assertNotIn('OTHER-99', str(result))
                    self.assertEqual(len(result['context']), 1)
                    self.assertEqual(result['context'][0]['metadata']['document_id'], document.pk)
                    self.assertEqual(judge.called, evaluate)
                    self.assertEqual(result['degraded'], [])

    def test_text_file_and_chunking_settings_reach_the_index(self):
        document = self.upload(text=('The launch code is ORCHID-72. ' * 30), config={
            'chunk_strategy': 'fixed', 'chunk_size': 100, 'chunk_overlap': 20,
            'embedding_model': 'custom/embedding'})
        collection = UserCollection.objects.get(user=self.user)
        self.assertEqual(collection.embedding_model, 'custom/embedding')
        stored = self.chroma.get_collection(collection.collection_name).get()
        self.assertGreater(len(stored['documents']), 5)
        self.assertTrue(all(len(text) <= 100 for text in stored['documents']))
        self.assertEqual(set(stored['ids']), set(document.document_chunks.values_list('chroma_id', flat=True)))
        self.query(embedding_model='different/model')
        self.assertEqual(self.provider.embedding_calls[-1]['model'], 'custom/embedding')

    def test_deletion_removes_vectors_and_allows_a_new_embedding_model(self):
        document = self.upload()
        collection = UserCollection.objects.get(user=self.user)
        response = self.client.delete(f'/api/v1/document/{document.pk}/alice/')
        self.assertEqual(response.status_code, 200, response.content)
        self.assertEqual([c.name for c in self.chroma.list_collections()], ['ragreader_collection'])
        collection.refresh_from_db()
        self.assertEqual((collection.chunk_count, collection.embedding_model), (0, ''))
        self.upload(config={'embedding_model': 'new/embedding'})
        collection.refresh_from_db()
        self.assertEqual(collection.embedding_model, 'new/embedding')

    def test_failed_upload_can_be_resubmitted_and_status_is_visible(self):
        response = self.client.post('/api/v1/insert-text/', {'USER': 'alice', 'TEXT': 'retry this'}, content_type='application/json')
        document = Document.objects.get(pk=response.json()['data']['document_id'])
        document.status = 'failed'
        document.error_message = 'Temporary provider failure'
        document.save()
        status = self.client.get(f'/api/v1/document/{document.pk}/alice/').json()['data']
        self.assertEqual(status['error_message'], 'Temporary provider failure')
        again = self.client.post('/api/v1/insert-text/', {'USER': 'alice', 'TEXT': 'retry this'}, content_type='application/json')
        self.assertEqual(again.json()['data']['document_id'], document.pk)
        self.assertEqual(again.json()['data']['status'], 'pending')
        self.assertEqual(self.dispatch.call_count, 2)
        build_index_task(**self.dispatch.call_args.kwargs)
        document.refresh_from_db()
        self.assertEqual(document.status, 'ready')

    def test_pending_upload_never_falls_back_to_the_base_corpus(self):
        self.client.post('/api/v1/insert-text/', {'USER': 'alice', 'TEXT': 'pending'}, content_type='application/json')
        result = self.query(corpus='auto').json()['data']
        self.assertEqual(result['source'], 'user_collection_indexing')
        self.assertEqual(result['context'], [])
        self.assertFalse(self.provider.chat_calls)

    def test_another_users_document_cannot_be_polled_or_deleted(self):
        document = self.upload(username='bob')
        self.assertEqual(self.client.get(f'/api/v1/document/{document.pk}/alice/').status_code, 404)
        self.assertEqual(self.client.delete(f'/api/v1/document/{document.pk}/alice/').status_code, 404)
        self.assertTrue(Document.objects.filter(pk=document.pk).exists())

    def test_no_readable_text_is_rejected_before_enqueuing(self):
        for file in (SimpleUploadedFile('empty.txt', b'  \n  '), SimpleUploadedFile('broken.pdf', b'not a pdf')):
            response = self.client.post('/api/v1/insert-data/', {'USER': 'alice', 'FILE': file})
            self.assertEqual(response.status_code, 400)
        self.assertFalse(Document.objects.exists())
        self.dispatch.assert_not_called()

    def test_text_starting_with_a_url_is_stored_without_fetching_it(self):
        with mock.patch.object(DataLoader, '_fetch_url') as fetch:
            self.upload('https://example.test is written in my notes.pdf')
        fetch.assert_not_called()

    def test_corpus_settings_reflect_indexing_and_deletion(self):
        url = '/api/v1/corpus/alice/'
        before = self.client.get(url)
        self.assertEqual(before.status_code, 200)
        self.assertFalse(before.json()['user_corpus']['locked'])
        document = self.upload(config={'embedding_model': 'custom/embedding'})
        after = self.client.get(url).json()['user_corpus']
        self.assertTrue(after['locked'])
        self.assertEqual(after['embedding_model'], 'custom/embedding')
        self.client.delete(f'/api/v1/document/{document.pk}/alice/')
        self.assertFalse(self.client.get(url).json()['user_corpus']['locked'])

    def test_plain_text_and_markdown_files_are_read_as_text(self):
        for name in ('notes.txt', 'notes.md'):
            with self.subTest(name=name):
                document = self.upload(file=SimpleUploadedFile(name, f'{name}: The launch code is ORCHID-72.'.encode()))
                self.assertEqual(document.source_type, 'text')
                self.assertEqual(document.name, name)

    def test_url_extraction_is_indexed_and_used_for_the_answer(self):
        with mock.patch.object(DataLoader, '_fetch_url', return_value='<main>The launch code is ORCHID-72.</main>'):
            response = self.client.post('/api/v1/insert-url/', {
                'USER': 'alice', 'URL': 'https://example.test/mission',
            }, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        build_index_task(**self.dispatch.call_args.kwargs)
        result = self.query().json()['data']
        self.assertIn('ORCHID-72', result['answer'])
        self.assertEqual(result['context'][0]['metadata']['source_type'], 'url')

    def test_redelivered_stale_indexing_job_recovers(self):
        response = self.client.post('/api/v1/insert-text/', {'USER': 'alice', 'TEXT': 'recover'}, content_type='application/json')
        pk = response.json()['data']['document_id']
        Document.objects.filter(pk=pk).update(status='indexing', updated_at=timezone.now() - timedelta(hours=1))
        build_index_task(**self.dispatch.call_args.kwargs)
        self.assertEqual(Document.objects.get(pk=pk).status, 'ready')

    def test_concurrent_requests_have_separate_collections_and_emitters(self):
        # A barrier forces both chains to switch collections before retrieving.
        # This reproduces the former shared-state race without threaded DB access.
        import threading
        barrier = threading.Barrier(2)
        scopes = [self.pipeline._request_pipeline(), self.pipeline._request_pipeline()]
        def retrieve(index):
            scope = scopes[index]
            scope.dense_rag.collection = f'user-{index}'
            scope.dense_multi_hop.set_emitter(f'emitter-{index}')
            barrier.wait(timeout=5)
            return scope.dense_multi_hop.retriever.retriever.collection, scope.dense_rag.emitter
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(retrieve, (0, 1)))
        self.assertEqual(result, [('user-0', 'emitter-0'), ('user-1', 'emitter-1')])
