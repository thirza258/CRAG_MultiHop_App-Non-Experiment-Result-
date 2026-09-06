"""Record the embedding model each existing collection was really built with.

``UserCollection.embedding_model`` and ``ChromaCollection.embedding_model`` have
existed since their tables did, but nothing ever wrote them — every row still
carries the field default, ``"text-embedding-3-small"``, while the vectors in
those collections were produced by whatever ``dense_config["embedding_model"]``
was set to in ``rag/rag_service.py`` and ``insert_base_dataset.py``. Both have
always been ``google/gemini-embedding-2-preview``.

That did not matter while the embedding model was a constant. It matters now:
query-time dense retrieval pins itself to the recorded value so a user's model
choice cannot embed a query into a different vector space than the collection it
searches. A recorded value that is wrong is worse than none at all, so the rows
are corrected here.

Only rows still holding the untouched default are changed. Anything else was
written deliberately (by code that post-dates this migration) and is left alone.
"""

from django.db import migrations

#: The value the fields defaulted to and that no code ever set.
UNWRITTEN_DEFAULT = "text-embedding-3-small"

#: What every collection was actually indexed with.
HISTORIC_EMBEDDING_MODEL = "google/gemini-embedding-2-preview"

#: The shared corpus insert_base_dataset.py builds.
BASE_COLLECTION_NAME = "ragreader_collection"


def backfill(apps, schema_editor):
    UserCollection = apps.get_model("router", "UserCollection")
    ChromaCollection = apps.get_model("router", "ChromaCollection")

    updated = UserCollection.objects.filter(
        embedding_model=UNWRITTEN_DEFAULT
    ).update(embedding_model=HISTORIC_EMBEDDING_MODEL)
    print(f"  backfilled {updated} UserCollection row(s)")

    ChromaCollection.objects.filter(embedding_model=UNWRITTEN_DEFAULT).update(
        embedding_model=HISTORIC_EMBEDDING_MODEL
    )

    # The base corpus is created directly against ChromaDB by
    # insert_base_dataset.py, which never wrote a bookkeeping row for it. The
    # pipeline falls back to its own dense_config when the row is missing, but
    # an explicit row is what lets the model be changed without a code change.
    _, created = ChromaCollection.objects.get_or_create(
        collection_name=BASE_COLLECTION_NAME,
        defaults={
            "collection_type": "corpus",
            "embedding_model": HISTORIC_EMBEDDING_MODEL,
        },
    )
    if created:
        print(f"  registered the base corpus '{BASE_COLLECTION_NAME}'")


class Migration(migrations.Migration):

    dependencies = [
        ("router", "0016_document_unique_user_file_hash"),
    ]

    operations = [
        # Irreversible by nature: reversing cannot restore a value that was
        # never recorded in the first place, and re-writing the wrong default
        # would put the misinformation back.
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
