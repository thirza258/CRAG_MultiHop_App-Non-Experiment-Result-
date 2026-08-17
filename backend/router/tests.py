"""API tests for the router endpoints, all mounted under /api/v1/.

Covers sign-up, the three insert endpoints (including the file_hash dedup
path), document listing, the two conversation read endpoints, /query/ and the
delete route the frontend calls. Every external dependency is patched out --
the DataLoader (get_loader), the celery dispatch (build_index_task.delay),
ChromaDB (get_chroma_client) and the RAG registry (get_registry) -- so the
whole module runs with no network, no API key and no Redis/Postgres/ChromaDB.

A number of tests deliberately pin CURRENT, wrong behaviour: those views
swallow their own failures, so the honest way to protect them is to assert what
they really return today. Each such test carries a "KNOWN BUG" comment naming
the defect, and is written so it fails loudly the moment the bug is fixed.
"""

import unittest
from datetime import timedelta
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.utils import timezone

try:
    from router import views as router_views
    from router.models import (
        Conversation,
        ConversationHistory,
        Document,
        DocumentChunk,
        GuestUser,
        UserCollection,
    )
    from authentication import views as auth_views
    from common.pipeline_config import DEFAULT_PIPELINE_CONFIG, MAX_HOPS
    from pipeline.app_pipeline import AppRAGPipeline
    IMPORT_ERROR = ""
except Exception as exc:  # needs Django + the heavy pipeline import chain
    router_views = None
    auth_views = None
    AppRAGPipeline = None
    DEFAULT_PIPELINE_CONFIG = {}
    MAX_HOPS = 0
    IMPORT_ERROR = str(exc)


API = "/api/v1"

@unittest.skipIf(router_views is None, f"router.views unavailable: {IMPORT_ERROR}")
class SignUpAPITests(TestCase):
    """POST /api/v1/sign-up/ (authentication.views.SignUpView)."""

    URL = f"{API}/sign-up/"

    def setUp(self):
        # SignUpView imports create_chroma_collection but never calls it today;
        # patched anyway so the test can never reach a ChromaDB server.
        patcher = mock.patch.object(auth_views, "create_chroma_collection")
        self.create_chroma_collection = patcher.start()
        self.addCleanup(patcher.stop)

    def _post(self, **payload):
        return self.client.post(self.URL, payload, content_type="application/json")

    def test_sign_up_creates_a_guest_user(self):
        resp = self._post(EMAIL="bob@example.com", USERNAME="bob")

        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["status"], 201)
        self.assertEqual(
            body["data"],
            {
                "response": "User created successfully!",
                "username": "bob",
                "email": "bob@example.com",
            },
        )
        self.assertTrue(
            GuestUser.objects.filter(username="bob", email="bob@example.com").exists()
        )

    def test_sign_up_is_idempotent_for_a_repeat_username_and_email(self):
        first = self._post(EMAIL="bob@example.com", USERNAME="bob")
        second = self._post(EMAIL="bob@example.com", USERNAME="bob")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["data"]["response"], "User already exists")
        self.assertEqual(GuestUser.objects.filter(username="bob").count(), 1)

    def test_sign_up_with_a_taken_username_and_a_new_email_returns_500(self):
        self._post(EMAIL="bob@example.com", USERNAME="bob")

        resp = self._post(EMAIL="bob+typo@example.com", USERNAME="bob")

        # KNOWN BUG: get_or_create() matches on (email, username) together, so a
        # returning user who mistypes their email misses the lookup, hits the
        # unique-username constraint and gets an opaque 500 instead of a 409/400.
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(GuestUser.objects.filter(username="bob").count(), 1)
        self.assertFalse(
            GuestUser.objects.filter(email="bob+typo@example.com").exists()
        )

    def test_sign_up_rejects_a_malformed_email(self):
        resp = self._post(EMAIL="not-an-email", USERNAME="bob")

        # KNOWN BUG: is_valid(raise_exception=True) raises a DRF ValidationError
        # that the view's blanket `except Exception` swallows, so DRF never gets
        # to turn it into a 400. The rejection is correct, the status code is not.
        self.assertEqual(resp.status_code, 500)
        self.assertIn("EMAIL", resp.json()["message"])
        self.assertFalse(GuestUser.objects.exists())


@unittest.skipIf(router_views is None, f"router.views unavailable: {IMPORT_ERROR}")
class RouterAPITestCase(TestCase):
    """Shared fixture plus the patches that keep the router views offline."""

    def setUp(self):
        self.user = GuestUser.objects.create(
            email="alice@example.com", username="alice"
        )

        # DataLoader would write to MEDIA_ROOT, parse PDFs and fetch URLs.
        self.loader = mock.Mock()
        self.loader.process_input.return_value = {
            "user": "alice",
            "text": "extracted body text",
            "name": "alice-note",
            "filename": "notes.pdf",
            "source_type": "url",
            "source_path": "documents/alice/notes.pdf",
            "text_path": "documents/alice/extracted.txt",
        }
        self.get_loader = self._patch("get_loader", return_value=self.loader)

        # Celery dispatch, ChromaDB and the RAG engine.
        self.build_index_task = self._patch("build_index_task")
        self.get_chroma_client = self._patch("get_chroma_client")
        self.get_registry = self._patch("get_registry")
        self.engine = self.get_registry.return_value.get_engine.return_value

    def _patch(self, attribute, **kwargs):
        patcher = mock.patch.object(router_views, attribute, **kwargs)
        started = patcher.start()
        self.addCleanup(patcher.stop)
        return started


class InsertDataAPITests(RouterAPITestCase):
    """POST /api/v1/insert-data/ -- multipart upload."""

    URL = f"{API}/insert-data/"

    def _upload(self, content=b"%PDF-1.4 fake pdf bytes", name="notes.pdf",
                username="alice"):
        return self.client.post(
            self.URL,
            {
                "USER": username,
                "FILE": SimpleUploadedFile(name, content, content_type="application/pdf"),
            },
        )

    def test_upload_creates_a_document_and_dispatches_indexing(self):
        resp = self._upload()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"], "Data inserted successfully!")

        document = Document.objects.get()
        self.assertEqual(document.user, self.user)
        self.assertEqual(document.name, "notes.pdf")
        self.assertEqual(document.source_type, "pdf")
        self.assertEqual(document.status, "pending")
        self.assertEqual(document.source_path, "documents/alice/notes.pdf")
        self.assertEqual(document.extracted_text_path, "documents/alice/extracted.txt")
        self.assertEqual(len(document.file_hash), 64)  # sha256 hexdigest

        self.build_index_task.delay.assert_called_once_with(
            document_id=document.pk, username="alice"
        )

    def test_reposting_identical_content_neither_duplicates_nor_reindexes(self):
        first = self._upload(name="notes.pdf")
        second = self._upload(name="renamed.pdf")  # same bytes -> same file_hash

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(Document.objects.count(), 1)

        document = Document.objects.get()
        self.assertIn("already indexed", second.json()["data"])
        self.assertIn(f"id={document.pk}", second.json()["data"])
        self.build_index_task.delay.assert_called_once_with(
            document_id=document.pk, username="alice"
        )

        # KNOWN (minor) BUG: the loader runs before the dedup check, so a
        # duplicate upload still re-saves the file and re-parses the PDF.
        self.assertEqual(self.loader.process_input.call_count, 2)

    def test_different_content_creates_a_second_document(self):
        self._upload(content=b"first pdf bytes")
        self._upload(content=b"second, different pdf bytes")

        self.assertEqual(Document.objects.count(), 2)
        self.assertEqual(self.build_index_task.delay.call_count, 2)

    def test_missing_file_is_rejected_without_creating_or_dispatching(self):
        resp = self.client.post(self.URL, {"USER": "alice"})

        # KNOWN BUG: serializer validation should surface as a 400; the view's
        # blanket `except Exception` turns the DRF ValidationError into a 500.
        self.assertEqual(resp.status_code, 500)
        self.assertIn("FILE", resp.json()["message"])
        self.assertFalse(Document.objects.exists())
        self.build_index_task.delay.assert_not_called()

    def test_upload_for_an_unknown_user_creates_nothing(self):
        resp = self._upload(username="ghost")

        # KNOWN BUG: GuestUser.DoesNotExist is swallowed into a 500 instead of
        # the 404 the frontend can act on.
        self.assertEqual(resp.status_code, 500)
        self.assertFalse(Document.objects.exists())
        self.build_index_task.delay.assert_not_called()


class InsertTextAPITests(RouterAPITestCase):
    """POST /api/v1/insert-text/."""

    URL = f"{API}/insert-text/"

    def test_pasted_text_is_indexed(self):
        # Regression guard: the view used to read validated_data["FILE"], which
        # InsertTextSerializer does not declare, so every paste raised
        # KeyError('FILE') and became a 500. /insert-text/ could never index.
        resp = self.client.post(
            self.URL,
            {"USER": "alice", "TEXT": "some pasted notes"},
            content_type="application/json",
        )

        self.assertEqual(resp.status_code, 200)
        # The pasted text itself is what gets handed to the loader.
        self.loader.process_input.assert_called_once_with("some pasted notes", "alice")

        document = Document.objects.get()
        self.assertEqual(document.source_type, "text")
        self.assertEqual(document.status, "pending")
        self.build_index_task.delay.assert_called_once_with(
            document_id=document.pk, username="alice"
        )

    def test_reposting_identical_text_neither_duplicates_nor_reindexes(self):
        payload = {"USER": "alice", "TEXT": "some pasted notes"}
        self.client.post(self.URL, payload, content_type="application/json")
        self.build_index_task.delay.reset_mock()

        resp = self.client.post(self.URL, payload, content_type="application/json")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Document.objects.count(), 1)
        self.build_index_task.delay.assert_not_called()

    def test_different_text_creates_a_second_document(self):
        self.client.post(
            self.URL,
            {"USER": "alice", "TEXT": "first note"},
            content_type="application/json",
        )
        self.client.post(
            self.URL,
            {"USER": "alice", "TEXT": "second note"},
            content_type="application/json",
        )

        self.assertEqual(Document.objects.count(), 2)

    def test_missing_text_field_is_rejected(self):
        resp = self.client.post(
            self.URL, {"USER": "alice"}, content_type="application/json"
        )

        # KNOWN BUG: should be a 400 (see InsertDataAPITests).
        self.assertEqual(resp.status_code, 500)
        self.assertIn("TEXT", resp.json()["message"])
        self.assertFalse(Document.objects.exists())
        self.build_index_task.delay.assert_not_called()


class InsertURLAPITests(RouterAPITestCase):
    """POST /api/v1/insert-url/."""

    URL = f"{API}/insert-url/"

    def test_a_submitted_url_is_indexed(self):
        # Regression guard: the view used to read validated_data["FILE"] while
        # InsertURLSerializer declares USER/URL, so every URL became a 500.
        resp = self.client.post(
            self.URL,
            {"USER": "alice", "URL": "https://example.com/article"},
            content_type="application/json",
        )

        self.assertEqual(resp.status_code, 200)
        self.loader.process_input.assert_called_once_with(
            "https://example.com/article", "alice"
        )

        document = Document.objects.get()
        self.assertEqual(document.source_type, "url")
        self.build_index_task.delay.assert_called_once_with(
            document_id=document.pk, username="alice"
        )

    def test_resubmitting_the_same_url_neither_duplicates_nor_reindexes(self):
        payload = {"USER": "alice", "URL": "https://example.com/article"}
        self.client.post(self.URL, payload, content_type="application/json")
        self.build_index_task.delay.reset_mock()

        resp = self.client.post(self.URL, payload, content_type="application/json")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Document.objects.count(), 1)
        self.build_index_task.delay.assert_not_called()

    def test_a_different_url_creates_a_second_document(self):
        for path in ("first", "second"):
            self.client.post(
                self.URL,
                {"USER": "alice", "URL": f"https://example.com/{path}"},
                content_type="application/json",
            )

        self.assertEqual(Document.objects.count(), 2)

    def test_malformed_url_is_rejected_by_the_serializer(self):
        resp = self.client.post(
            self.URL,
            {"USER": "alice", "URL": "not-a-url"},
            content_type="application/json",
        )

        # KNOWN BUG: should be a 400. The distinguishing detail from the test
        # above is the message -- this one is real field validation, not the
        # KeyError('FILE').
        self.assertEqual(resp.status_code, 500)
        self.assertIn("valid URL", resp.json()["message"])
        self.assertFalse(Document.objects.exists())


class DocumentListAPITests(RouterAPITestCase):
    """GET /api/v1/document/<username>/."""

    def setUp(self):
        super().setUp()
        self.document = Document.objects.create(
            user=self.user,
            name="notes.pdf",
            source_type="pdf",
            source_path="documents/alice/notes.pdf",
            extracted_text_path="documents/alice/extracted.txt",
            file_hash="a" * 64,
            status="ready",
        )
        other = GuestUser.objects.create(email="bob@example.com", username="bob")
        Document.objects.create(
            user=other,
            name="bob.pdf",
            source_type="pdf",
            source_path="documents/bob/bob.pdf",
            extracted_text_path="documents/bob/extracted.txt",
            file_hash="b" * 64,
        )

    def test_lists_only_the_requested_users_documents(self):
        resp = self.client.get(f"{API}/document/alice/")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(len(data), 1)

        entry = data[0]
        self.assertEqual(entry["id"], self.document.pk)
        self.assertEqual(entry["name"], "notes.pdf")
        self.assertEqual(entry["source_type"], "pdf")
        self.assertEqual(entry["source_path"], "documents/alice/notes.pdf")
        self.assertEqual(entry["extracted_text_path"], "documents/alice/extracted.txt")
        self.assertIn("created_at", entry)

    def test_extracted_text_path_is_truncated_to_100_characters(self):
        long_path = "documents/alice/" + "x" * 200
        Document.objects.filter(pk=self.document.pk).update(
            extracted_text_path=long_path
        )

        resp = self.client.get(f"{API}/document/alice/")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"][0]["extracted_text_path"], long_path[:100])

    def test_unknown_user_returns_a_404(self):
        # Regression guard: common.schema.RAGResponse.response_404 used to lack
        # @staticmethod, so calling it on the get_responses() instance bound the
        # instance into its `error` parameter and raised TypeError. The blanket
        # `except Exception` then turned every intended 404 into a 500.
        resp = self.client.get(f"{API}/document/ghost/")

        self.assertEqual(resp.status_code, 404)
        self.assertIn("User not found", resp.json()["message"])

    def test_user_with_no_documents_returns_a_404(self):
        GuestUser.objects.create(email="carol@example.com", username="carol")

        resp = self.client.get(f"{API}/document/carol/")

        self.assertEqual(resp.status_code, 404)
        self.assertIn("Document not found", resp.json()["message"])


class ConversationAPITests(RouterAPITestCase):
    """GET /api/v1/conversation/<id>/."""

    def test_returns_the_stored_conversation(self):
        conversation = Conversation.objects.create(
            user=self.user,
            query="who wrote it?",
            response="Ada Lovelace wrote it.",
            context="chunk one\n\nchunk two",
        )

        resp = self.client.get(f"{API}/conversation/{conversation.pk}/")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["id"], conversation.pk)
        self.assertEqual(data["query"], "who wrote it?")
        self.assertEqual(data["response"], "Ada Lovelace wrote it.")
        self.assertEqual(data["context"], "chunk one\n\nchunk two")
        self.assertIn("created_at", data)

    def test_non_numeric_conversation_id_returns_the_json_error_envelope(self):
        resp = self.client.get(f"{API}/conversation/abc/")

        # The ValueError is raised in the try body, so the blanket
        # `except Exception` catches it and the app's JSON shape survives.
        # Contrast with the missing-id test below.
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp["Content-Type"], "application/json")
        self.assertEqual(resp.json()["status"], 500)

    def test_missing_conversation_returns_the_json_404_envelope(self):
        # Regression guard: response_404 is called from inside
        # `except Conversation.DoesNotExist`, so while it lacked @staticmethod its
        # TypeError escaped the view entirely and the client got Django's generic
        # HTML 500 instead of the API's JSON error envelope.
        client = Client(raise_request_exception=False)

        resp = client.get(f"{API}/conversation/999999/")

        self.assertEqual(resp.status_code, 404)
        self.assertIn("application/json", resp["Content-Type"])


class ConversationHistoryAPITests(RouterAPITestCase):
    """GET /api/v1/conversation-history/<username>/."""

    def test_returns_the_users_history_newest_first(self):
        older = Conversation.objects.create(
            user=self.user, query="first?", response="one"
        )
        newer = Conversation.objects.create(
            user=self.user, query="second?", response="two"
        )
        older_history = ConversationHistory.objects.create(
            user=self.user, conversation=older
        )
        newer_history = ConversationHistory.objects.create(
            user=self.user, conversation=newer
        )

        # auto_now_add can stamp both rows inside the same microsecond, which
        # would make the -created_at ordering assertion flaky.
        now = timezone.now()
        ConversationHistory.objects.filter(pk=older_history.pk).update(
            created_at=now - timedelta(minutes=5)
        )
        ConversationHistory.objects.filter(pk=newer_history.pk).update(created_at=now)

        bob = GuestUser.objects.create(email="bob@example.com", username="bob")
        bob_conversation = Conversation.objects.create(
            user=bob, query="bob?", response="not alice's"
        )
        ConversationHistory.objects.create(user=bob, conversation=bob_conversation)

        resp = self.client.get(f"{API}/conversation-history/alice/")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual([row["query"] for row in data], ["second?", "first?"])
        self.assertEqual([row["response"] for row in data], ["two", "one"])
        self.assertTrue(all("created_at" in row for row in data))

    def test_user_with_no_history_returns_an_empty_list(self):
        resp = self.client.get(f"{API}/conversation-history/alice/")

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"], [])

    def test_unknown_user_returns_the_json_404_envelope(self):
        # Same shape as ConversationAPITests: response_404 is called from inside
        # `except GuestUser.DoesNotExist`, so its TypeError used to escape the
        # view instead of becoming the API's JSON 404.
        client = Client(raise_request_exception=False)

        resp = client.get(f"{API}/conversation-history/ghost/")

        self.assertEqual(resp.status_code, 404)
        self.assertIn("application/json", resp["Content-Type"])


class QueryAPITests(RouterAPITestCase):
    """POST /api/v1/query/."""

    URL = f"{API}/query/"

    ENGINE_RESULT = {
        "answer": "Ada Lovelace wrote it.",
        "source": "user_collection",
        "context": [
            {"text": "chunk one", "metadata": {"document_id": 1}},
            {"text": "chunk two", "metadata": {"document_id": 1}},
        ],
        "evaluation": {"answer_relevancy": None, "faithfulness": None},
        "degraded": [],
    }

    def _post(self, payload=None):
        return self.client.post(
            self.URL,
            payload if payload is not None else {"USER": "alice", "QUERY": "who wrote it?"},
            content_type="application/json",
        )

    def test_engine_payload_is_returned_with_a_conversation_id(self):
        self.engine.run.return_value = dict(self.ENGINE_RESULT)

        resp = self._post()

        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["answer"], "Ada Lovelace wrote it.")
        self.assertEqual(data["source"], "user_collection")
        self.assertEqual(data["degraded"], [])
        self.assertEqual(
            data["evaluation"], {"answer_relevancy": None, "faithfulness": None}
        )
        self.assertEqual(
            data["context"],
            [
                {"text": "chunk one", "metadata": {"document_id": 1}},
                {"text": "chunk two", "metadata": {"document_id": 1}},
            ],
        )

        placeholder = Conversation.objects.get(response="")
        self.assertEqual(self.engine.run.call_count, 1)
        args, kwargs = self.engine.run.call_args
        self.assertEqual(args, ("alice", "who wrote it?"))
        self.assertEqual(kwargs["conversation_id"], placeholder.pk)
        self.assertEqual(kwargs["config"], DEFAULT_PIPELINE_CONFIG)

        answer_record = Conversation.objects.get(pk=data["conversation_id"])
        self.assertEqual(answer_record.user, self.user)
        self.assertEqual(answer_record.query, "who wrote it?")
        self.assertEqual(answer_record.response, "Ada Lovelace wrote it.")
        self.assertEqual(answer_record.context, "chunk one\n\nchunk two")
        self.assertEqual(
            ConversationHistory.objects.filter(conversation=answer_record).count(), 1
        )

    def test_a_config_payload_is_normalised_before_reaching_the_engine(self):
        self.engine.run.return_value = dict(self.ENGINE_RESULT)

        resp = self._post(
            {
                "USER": "alice",
                "QUERY": "who wrote it?",
                "CONFIG": {
                    "use_reranker": False,
                    "max_hops": 99,
                    "retrievers": "dense",
                    "bogus": "dropped",
                },
            }
        )

        self.assertEqual(resp.status_code, 200)
        config = self.engine.run.call_args[1]["config"]
        self.assertFalse(config["use_reranker"])
        self.assertEqual(config["max_hops"], MAX_HOPS)  # 99 is clamped, not rejected
        self.assertEqual(config["retrievers"], "dense")
        self.assertNotIn("bogus", config)
        self.assertTrue(config["use_multi_hop"])  # untouched keys keep the default

    def test_a_malformed_config_falls_back_to_the_defaults(self):
        self.engine.run.return_value = dict(self.ENGINE_RESULT)

        resp = self._post(
            {
                "USER": "alice",
                "QUERY": "who wrote it?",
                "CONFIG": {"max_hops": "not-a-number", "corpus": "nonsense"},
            }
        )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.engine.run.call_args[1]["config"], DEFAULT_PIPELINE_CONFIG)

    def test_every_query_also_stores_an_empty_placeholder_conversation(self):
        self.engine.run.return_value = dict(self.ENGINE_RESULT)

        resp = self._post()

        # KNOWN BUG: save_conversation() is called once before the engine runs
        # (to mint the conversation_id the pipeline streams status under) and
        # again with the answer, so every query writes two Conversation rows and
        # two ConversationHistory rows. Consequences: the blank placeholder shows
        # up as an empty entry in /conversation-history/, and the
        # conversation_id handed back to the browser is NOT the one the pipeline
        # published its status events under.
        self.assertEqual(Conversation.objects.count(), 2)
        self.assertEqual(ConversationHistory.objects.count(), 2)

        placeholder = Conversation.objects.get(response="")
        self.assertEqual(
            ConversationHistory.objects.filter(conversation=placeholder).count(), 1
        )
        self.assertNotEqual(
            resp.json()["data"]["conversation_id"],
            self.engine.run.call_args[1]["conversation_id"],
        )

    def test_engine_failure_returns_the_json_error_envelope(self):
        self.engine.run.side_effect = RuntimeError("chroma down")

        resp = self._post()

        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp["Content-Type"], "application/json")
        body = resp.json()
        self.assertEqual(body["status"], 500)
        self.assertEqual(body["message"], "chroma down")
        self.assertIsNone(body["data"])

    @unittest.skipIf(
        AppRAGPipeline is None, f"pipeline unavailable: {IMPORT_ERROR}"
    )
    def test_the_engine_call_matches_the_real_pipeline_run_signature(self):
        # A bare Mock engine accepts any argument list, so it cannot catch an
        # arity regression -- and there was one: the view called
        # run(username, query) while AppRAGPipeline.run requires
        # conversation_id, so every HTTP query 500'd. create_autospec enforces
        # the real signature. (router/tasks.py:37 still has the short call.)
        engine = mock.create_autospec(AppRAGPipeline, instance=True)
        engine.run.return_value = dict(self.ENGINE_RESULT)
        self.get_registry.return_value.get_engine.return_value = engine

        resp = self._post()

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["data"]["answer"], "Ada Lovelace wrote it.")
        engine.run.assert_called_once()
        # Guards against the autospec silently degrading into a tautology.
        self.assertEqual(
            engine.run.call_args[1]["conversation_id"],
            Conversation.objects.get(response="").pk,
        )

    def test_missing_query_field_is_rejected_before_the_engine_runs(self):
        resp = self._post({"USER": "alice"})

        # KNOWN BUG: should be a 400 (see InsertDataAPITests).
        self.assertEqual(resp.status_code, 500)
        self.assertIn("QUERY", resp.json()["message"])
        self.engine.run.assert_not_called()
        self.assertEqual(Conversation.objects.count(), 0)

    def test_unknown_user_is_rejected_before_the_engine_runs(self):
        resp = self._post({"USER": "ghost", "QUERY": "who wrote it?"})

        # KNOWN BUG: GuestUser.DoesNotExist -> 500 rather than 404.
        self.assertEqual(resp.status_code, 500)
        self.engine.run.assert_not_called()
        self.assertEqual(Conversation.objects.count(), 0)


class DeleteDocumentRouteTests(RouterAPITestCase):
    """The delete route the sidebar calls."""

    def setUp(self):
        super().setUp()
        self.document = Document.objects.create(
            user=self.user,
            name="notes.pdf",
            source_type="pdf",
            source_path="documents/alice/notes.pdf",
            extracted_text_path="documents/alice/extracted.txt",
            file_hash="a" * 64,
        )

    def test_the_document_the_sidebar_asks_to_delete_is_deleted(self):
        # Regression guard: DeleteDocumentView existed and matched the signature
        # frontend/src/services/service.ts calls, but router/urls.py never routed
        # it, so every delete from the UI 404'd and nothing was ever removed.
        UserCollection.objects.create(
            user=self.user, collection_name="user_alice_collection"
        )

        resp = self.client.delete(f"{API}/document/{self.document.pk}/alice/")

        self.assertEqual(resp.status_code, 200)
        self.assertFalse(Document.objects.filter(pk=self.document.pk).exists())

    def test_deleting_an_unknown_document_reports_not_found(self):
        resp = self.client.delete(f"{API}/document/999999/alice/")

        self.assertEqual(resp.status_code, 404)
        self.assertTrue(Document.objects.filter(pk=self.document.pk).exists())

    def test_a_documents_chunks_are_removed_from_the_vector_store(self):
        collection = UserCollection.objects.create(
            user=self.user, collection_name="user_alice_collection", chunk_count=2
        )
        for index in range(2):
            DocumentChunk.objects.create(
                document=self.document,
                user_collection=collection,
                chroma_id=f"alice_{self.document.pk}_chunk_{index}",
                chunk_index=index,
            )

        resp = self.client.delete(f"{API}/document/{self.document.pk}/alice/")

        self.assertEqual(resp.status_code, 200)
        # The chunk ids, not the whole collection, are what gets dropped.
        self.get_chroma_client.return_value.delete.assert_called_once()
        deleted_ids = self.get_chroma_client.return_value.delete.call_args.kwargs["ids"]
        self.assertEqual(len(deleted_ids), 2)
        self.assertFalse(
            DocumentChunk.objects.filter(document_id=self.document.pk).exists()
        )
        collection.refresh_from_db()
        self.assertEqual(collection.chunk_count, 0)

    def test_delete_on_the_document_list_route_is_not_allowed(self):
        # The one nearby route that does resolve only implements GET. Accept is
        # pinned to JSON because this is the only response in the module that
        # goes through DRF's renderers rather than being a JsonResponse.
        resp = self.client.delete(
            f"{API}/document/alice/", HTTP_ACCEPT="application/json"
        )

        self.assertEqual(resp.status_code, 405)
        self.assertTrue(Document.objects.filter(pk=self.document.pk).exists())


if __name__ == "__main__":
    unittest.main()
