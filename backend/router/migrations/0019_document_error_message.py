from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("router", "0018_blank_embedding_model_default")]
    operations = [
        migrations.AddField(
            model_name="document",
            name="error_message",
            field=models.TextField(blank=True, default=""),
        ),
    ]
