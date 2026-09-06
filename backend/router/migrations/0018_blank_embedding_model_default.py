"""Stop the embedding-model fields from defaulting to a model name.

``"text-embedding-3-small"`` was the field default on both collection models
while nothing ever wrote them, so every row asserted a model that had not
produced its vectors. 0017 corrected the existing rows; this stops new ones from
being created with the same claim.

Blank now means "not recorded", which is what query-time pinning needs to be
able to express: dense retrieval pins itself to this value, so it has to be able
to tell "built by model X" from "we do not know".

Schema-only — existing values are untouched.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("router", "0017_backfill_collection_embedding_models"),
    ]

    operations = [
        migrations.AlterField(
            model_name="chromacollection",
            name="embedding_model",
            field=models.CharField(blank=True, default="", max_length=100),
        ),
        migrations.AlterField(
            model_name="usercollection",
            name="embedding_model",
            field=models.CharField(blank=True, default="", max_length=100),
        ),
    ]
