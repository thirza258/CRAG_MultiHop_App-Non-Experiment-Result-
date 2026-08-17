"""Strict import / wiring smoke test for the whole backend.

What it covers:

1. Every Python module under the sixteen backend packages imports cleanly.
   All failures are collected and reported together, so one broken module does
   not hide the other nine.
2. The root URLconf loads, and every named route the project itself owns
   reverses with dummy kwargs (the ``admin`` namespace is reversed via
   ``admin:index`` only — see ``SKIPPED_URL_NAMESPACES``).
3. Every websocket route in ``router.urls.websocket_urlpatterns`` resolves to a
   real ``AsyncWebsocketConsumer`` subclass, every ``self.x(...)`` it calls is
   defined, and every ``engine.x(...)`` it calls exists on the RAG engine class
   with a compatible signature.
4. ``manage.py check`` reports no errors.

This file deliberately contains **no** ``skipIf``. Every other test module in
this repo guards its heavy imports so it can skip in a thin environment; that
is how a dead websocket consumer and an unimplemented engine method survived.
This module is the counterweight: if the backend cannot be imported or wired,
it fails loudly with the module name and the exception.

Nothing here needs the network, an API key, Redis, PostgreSQL or a ChromaDB
server: no test constructs a retriever, opens a socket or touches the database.
"""

import ast
import importlib
import inspect
import io
import os
import pkgutil
import unittest
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ragreader.settings")

import django

# Idempotent: apps.populate() returns immediately when the registry is already
# ready, so this is a no-op under `manage.py test` and makes the module runnable
# under plain `python -m unittest` from the backend directory as well.
django.setup()

from channels.generic.websocket import AsyncWebsocketConsumer
from django.core.management import call_command
from django.urls import URLPattern, URLResolver, get_resolver, reverse

BACKEND_DIR = Path(__file__).resolve().parent

# The sixteen application packages. Root-level scripts (manage.py,
# dl_reranker_model.py, insert_base_dataset.py) are intentionally absent: they
# call django.setup() / sys.exit() at module scope and are covered by
# test_scripts.py.
PACKAGES = (
    "ai_handler",
    "authentication",
    "chroma",
    "common",
    "corrective",
    "dense_rag",
    "emitter",
    "evaluation",
    "hybrid_rag",
    "multi_hop",
    "pipeline",
    "rag",
    "ragreader",
    "router",
    "sparse_rag",
    "utils",
)

# Generated / non-production code. Migrations are Django's business and are
# already exercised by `check`; __pycache__ is not source.
SKIPPED_DIR_NAMES = frozenset({"migrations", "__pycache__"})

# Modules whose import genuinely requires a live service, and therefore cannot
# be asserted on in an offline environment. Kept as an explicit constant so an
# exclusion can never be silent.
#
# Audited every module-level side effect in the sixteen packages; the result is
# that nothing needs excluding:
#   * chroma.chroma_settings only *defines* the chromadb.HttpClient calls.
#   * rag.rag_service builds RAGRegistry() at import, but initialize_engine()
#     wraps AppRAGPipeline construction in try/except and only prints, so a
#     dead ChromaDB/absent API key cannot fail the import.
#   * ragreader.asgi / ragreader.wsgi build their ASGI/WSGI apps at import;
#     neither opens a connection.
#   * common.dataset_settings builds a DocumentChunker at import; that
#     constructor only stores its arguments.
#   * corrective.corrective_evaluator calls ensure_nltk_data() unguarded at
#     import, which downloads from the NLTK CDN when the corpora are missing.
#     Considered and NOT excluded: pipeline.app_pipeline -> rag.rag_service ->
#     router.consumers -> router.urls all pull it in transitively, so excluding
#     it here would buy nothing while hiding a failure that also stops Django
#     from booting. If it fails, that is a real finding, not a test defect.
EXCLUDED = frozenset()

# Reversing the whole django.contrib.admin namespace needs per-model kwargs
# (content type ids, real object ids) that a smoke test cannot invent. The
# namespace is covered by reversing admin:index instead.
SKIPPED_URL_NAMESPACES = frozenset({"admin"})

# The routes the frontend depends on. Asserted as a subset of what the walker
# discovers so that a refactor which drops a route (or breaks the walker) fails
# instead of vacuously passing over an empty list.
EXPECTED_ROUTE_NAMES = frozenset({
    "swagger-ui",
    "schema",
    "redoc",
    "sign-up",
    "insert-data",
    "insert-text",
    "insert-url",
    "document-detail",
    "conversation-history",
    "conversation",
    "query",
    "chunk-create",
})

# Local variable names that hold the RAG engine inside a consumer, and the
# factory calls that return it (router.consumers uses both
# `engine = rag_registry.get_engine(...)` and `get_registry().get_engine()`).
ENGINE_LOCAL_NAMES = frozenset({"engine", "rag_engine"})
ENGINE_FACTORY_NAMES = frozenset({"get_engine"})


def _is_test_module(dotted_name: str) -> bool:
    leaf = dotted_name.rpartition(".")[2]
    return leaf == "tests" or leaf.startswith("test_")


def _discover_modules() -> list:
    """Every importable module under PACKAGES, minus tests/migrations/EXCLUDED."""
    found = []
    for package in PACKAGES:
        found.append(package)
        stack = [(BACKEND_DIR / package, f"{package}.")]
        while stack:
            directory, prefix = stack.pop()
            for info in pkgutil.iter_modules([str(directory)]):
                if info.name in SKIPPED_DIR_NAMES:
                    continue
                dotted = prefix + info.name
                found.append(dotted)
                if info.ispkg:
                    stack.append((directory / info.name, f"{dotted}."))
    return sorted(
        name for name in set(found)
        if not _is_test_module(name) and name not in EXCLUDED
    )


class ModuleImportSmokeTests(unittest.TestCase):
    def test_module_discovery_finds_the_whole_backend(self):
        modules = _discover_modules()

        # A broken walker would make the import test below vacuous, so pin the
        # shape of what it must find: every package plus known-awkward members
        # (namespace packages with no __init__.py, and the deepest imports).
        for expected in (
            *PACKAGES,
            "chroma.chroma_settings",       # namespace package, no __init__.py
            "emitter.status",
            "pipeline.app_pipeline",
            "router.consumers",
            "ragreader.asgi",
            "utils.insert_file",
        ):
            self.assertIn(expected, modules)

        # Named explicitly rather than re-applying the walker's own predicates:
        # these files all exist on disk, so a skip rule that stops working (or
        # gets rewritten) surfaces here instead of silently re-importing tests.
        for excluded in (
            "ai_handler.tests",
            "authentication.tests",
            "common.tests",
            "common.test_pipeline_config",
            "evaluation.tests",
            "hybrid_rag.tests",
            "rag.tests",
            "router.tests",
            "router.migrations",
            "router.migrations.0001_initial",
            "evaluation.migrations.0001_initial",
        ):
            self.assertNotIn(excluded, modules)

    def test_every_backend_module_imports(self):
        failures = []
        for name in _discover_modules():
            try:
                importlib.import_module(name)
            except (Exception, SystemExit) as exc:
                # SystemExit is not an Exception: a module that calls sys.exit()
                # at import time must be recorded, not kill the whole run.
                failures.append(f"{name}: {type(exc).__name__}: {exc}")

        self.assertEqual(
            failures, [],
            "modules failed to import:\n  " + "\n  ".join(failures),
        )


def _dummy_for_converter(converter) -> str:
    """A reverse() argument that satisfies the path converter's own regex."""
    name = type(converter).__name__.lower()
    if "int" in name:
        return "1"
    if "uuid" in name:
        return "00000000-0000-0000-0000-000000000000"
    return "smoke"


def _pattern_group_names(pattern) -> list:
    try:
        return list(pattern.regex.groupindex)
    except Exception:
        # Django could rename the compiled-regex descriptor; the converter keys
        # are the same parameter names for every path()-declared route.
        return list(getattr(pattern, "converters", {}) or {})


def _iter_named_patterns(resolver, prefix=""):
    for entry in resolver.url_patterns:
        if isinstance(entry, URLResolver):
            namespace = entry.namespace
            if namespace in SKIPPED_URL_NAMESPACES:
                continue
            child_prefix = f"{prefix}{namespace}:" if namespace else prefix
            yield from _iter_named_patterns(entry, child_prefix)
        elif isinstance(entry, URLPattern) and entry.name:
            yield prefix + entry.name, entry


class UrlConfSmokeTests(unittest.TestCase):
    def test_root_urlconf_loads(self):
        # url_patterns is lazy: touching it is what imports ragreader.urls and,
        # through it, router.urls -> router.consumers -> the whole RAG stack.
        patterns = get_resolver().url_patterns
        self.assertTrue(patterns, "ROOT_URLCONF produced no url patterns")

    def test_every_project_named_route_reverses(self):
        discovered = {}
        for name, entry in _iter_named_patterns(get_resolver()):
            discovered[name] = entry

        missing = EXPECTED_ROUTE_NAMES - set(discovered)
        self.assertEqual(
            missing, set(),
            f"named routes missing from the URLconf: {sorted(missing)}",
        )

        for name, entry in sorted(discovered.items()):
            with self.subTest(route=name):
                converters = getattr(entry.pattern, "converters", {}) or {}
                kwargs = {
                    group: _dummy_for_converter(converters.get(group))
                    for group in _pattern_group_names(entry.pattern)
                }
                try:
                    resolved = reverse(name, kwargs=kwargs)
                except Exception as exc:  # noqa: BLE001
                    self.fail(
                        f"reverse({name!r}, kwargs={kwargs!r}) failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                self.assertTrue(resolved.startswith("/"))

    def test_admin_namespace_reverses(self):
        self.assertEqual(reverse("admin:index"), "/admin/")


def _consumer_class(entry):
    """Recover the class behind ``SomeConsumer.as_asgi()``.

    channels' as_asgi() returns a closure and stamps the class onto it as
    ``consumer_class``; functools.update_wrapper also leaves ``__wrapped__``.
    Both are probed so a channels change breaks one attribute, not the test.
    """
    callback = entry.callback
    return (
        getattr(callback, "consumer_class", None)
        or getattr(callback, "__wrapped__", None)
    )


def _class_def(cls):
    """The ast.ClassDef for ``cls``, read from the file it is defined in."""
    source_file = inspect.getsourcefile(cls)
    tree = ast.parse(Path(source_file).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return node, source_file
    return None, source_file


def _is_engine_expression(node) -> bool:
    if isinstance(node, ast.Name) and node.id in ENGINE_LOCAL_NAMES:
        return True
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ENGINE_FACTORY_NAMES
    )


def _engine_attribute_uses(class_def) -> dict:
    """``{attribute_name: first_line_number}`` for every engine.x reference.

    Covers both ``engine.x(...)`` and ``sync_to_async(engine.x)(...)``, which is
    how router.consumers actually reaches the engine from async code.
    """
    uses = {}
    for node in ast.walk(class_def):
        if isinstance(node, ast.Attribute) and _is_engine_expression(node.value):
            uses.setdefault(node.attr, node.lineno)
    return uses


def _engine_direct_calls(class_def) -> list:
    """``(attr, positional_count, keyword_names, lineno)`` for engine.x(...).

    Only direct calls: the argument list of a ``sync_to_async(engine.x)(...)``
    call belongs to the wrapper, so its arity cannot be attributed to x.
    """
    calls = []
    for node in ast.walk(class_def):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _is_engine_expression(node.func.value)
        ):
            calls.append((
                node.func.attr,
                len(node.args),
                sorted(kw.arg for kw in node.keywords if kw.arg),
                node.lineno,
            ))
    return calls


def _self_calls(class_def) -> dict:
    """``{method_name: lineno}`` for every ``self.x(...)`` in the class.

    Attribute *assignments* (self.job_id = ...) are excluded by matching calls
    only, so this flags genuinely missing methods rather than instance state.
    """
    calls = {}
    for node in ast.walk(class_def):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            calls.setdefault(node.func.attr, node.lineno)
    return calls


class WebsocketRouteSmokeTests(unittest.TestCase):
    def setUp(self):
        import router.urls

        self.websocket_patterns = list(router.urls.websocket_urlpatterns)
        # rag.rag_service.RAGRegistry.initialize_engine() builds exactly this
        # class, so it is the engine every consumer talks to.
        from pipeline.app_pipeline import AppRAGPipeline

        self.engine_class = AppRAGPipeline

    def test_websocket_routes_target_async_websocket_consumers(self):
        self.assertTrue(
            self.websocket_patterns,
            "router.urls.websocket_urlpatterns is empty — the ASGI app would "
            "accept no websocket connections at all",
        )

        for entry in self.websocket_patterns:
            with self.subTest(route=str(entry.pattern)):
                consumer = _consumer_class(entry)
                self.assertIsNotNone(
                    consumer,
                    f"could not recover the consumer class behind "
                    f"{entry.pattern} — neither .consumer_class nor "
                    f".__wrapped__ is set on {entry.callback!r}",
                )
                self.assertTrue(
                    isinstance(consumer, type)
                    and issubclass(consumer, AsyncWebsocketConsumer),
                    f"{entry.pattern} routes to {consumer!r}, which is not an "
                    f"AsyncWebsocketConsumer subclass",
                )

    def test_consumers_define_every_method_they_call_on_themselves(self):
        for entry in self.websocket_patterns:
            consumer = _consumer_class(entry)
            if consumer is None:
                continue  # already reported by the route test above
            class_def, source_file = _class_def(consumer)
            with self.subTest(consumer=consumer.__name__):
                self.assertIsNotNone(
                    class_def,
                    f"no class definition for {consumer.__name__} in {source_file}",
                )
                undefined = {
                    name: lineno
                    for name, lineno in _self_calls(class_def).items()
                    if not hasattr(consumer, name)
                }
                self.assertEqual(
                    undefined, {},
                    f"{consumer.__name__} calls self.<name>() for names it does "
                    f"not define and does not inherit: {sorted(undefined)} "
                    f"(see {source_file})",
                )

    def test_consumers_only_call_engine_methods_that_exist(self):
        for entry in self.websocket_patterns:
            consumer = _consumer_class(entry)
            if consumer is None:
                continue
            class_def, source_file = _class_def(consumer)
            if class_def is None:
                continue
            with self.subTest(consumer=consumer.__name__):
                uses = _engine_attribute_uses(class_def)
                missing = {
                    name: lineno
                    for name, lineno in uses.items()
                    if not hasattr(self.engine_class, name)
                }
                self.assertEqual(
                    missing, {},
                    f"{consumer.__name__} references "
                    + ", ".join(
                        f"engine.{name} (line {lineno})"
                        for name, lineno in sorted(missing.items())
                    )
                    + f" but {self.engine_class.__module__}."
                    f"{self.engine_class.__name__} defines none of them. Either "
                    f"implement them on the engine or delete the dead route and "
                    f"consumer ({source_file}, router/urls.py).",
                )

    def test_engine_signatures_accept_the_consumer_call_sites(self):
        for entry in self.websocket_patterns:
            consumer = _consumer_class(entry)
            if consumer is None:
                continue
            class_def, source_file = _class_def(consumer)
            if class_def is None:
                continue
            for attr, positional, keywords, lineno in _engine_direct_calls(class_def):
                target = getattr(self.engine_class, attr, None)
                if target is None:
                    continue  # reported by the missing-method test
                with self.subTest(consumer=consumer.__name__, call=attr):
                    signature = inspect.signature(target)
                    # target is the unbound function, so stand in for `self`.
                    args = [None] * (positional + 1)
                    kwargs = {name: None for name in keywords}
                    try:
                        signature.bind(*args, **kwargs)
                    except TypeError as exc:
                        self.fail(
                            f"{source_file}:{lineno} calls engine.{attr}() with "
                            f"{positional} positional arg(s) and keywords "
                            f"{keywords}, which {self.engine_class.__name__}."
                            f"{attr}{signature} rejects: {exc}"
                        )


class SystemCheckSmokeTests(unittest.TestCase):
    def test_django_system_check_reports_no_errors(self):
        output = io.StringIO()
        try:
            # No --database flag: the checks stay off the DB, so this passes
            # with no PostgreSQL reachable.
            call_command("check", stdout=output, stderr=output)
        except Exception as exc:  # noqa: BLE001 - SystemCheckError and friends
            self.fail(
                f"django system check failed: {type(exc).__name__}: {exc}\n"
                f"{output.getvalue()}"
            )


if __name__ == "__main__":
    unittest.main()
