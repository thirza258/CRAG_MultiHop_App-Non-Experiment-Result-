import json

from rest_framework import serializers


class JSONObjectField(serializers.Field):
    """A JSON object that may arrive as a JSON *string*.

    ``multipart/form-data`` has no notion of a nested object, so the upload
    endpoints receive CONFIG/KEYS as JSON text while the JSON endpoints receive
    real objects. Both spellings normalise to a dict here.

    Anything unparseable becomes ``{}`` rather than a validation error, matching
    the contract in ``common.runtime.config`` and ``common.runtime.api_keys``: a
    malformed settings blob falls back to the defaults instead of failing the
    request it was attached to.
    """

    def to_internal_value(self, data):
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
            except (TypeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def to_representation(self, value):
        return value


class QuerySerializer(serializers.Serializer):
    USER = serializers.CharField()
    QUERY = serializers.CharField()
    # Optional per-query pipeline selection; validated/clamped by
    # common.runtime.config, so anything shaped wrong falls back to the defaults.
    CONFIG = JSONObjectField(required=False)
    # The caller's own API keys, used for this request only and never stored.
    # Deliberately separate from CONFIG, which is logged verbatim — see
    # common.runtime.api_keys.
    KEYS = JSONObjectField(required=False)


class InsertURLSerializer(serializers.Serializer):
    USER = serializers.CharField()
    URL = serializers.URLField()
    # Indexing is where the embedding-model choice actually takes effect: a
    # collection's vectors all have to come from one model.
    CONFIG = JSONObjectField(required=False)
    KEYS = JSONObjectField(required=False)


class InsertDataSerializer(serializers.Serializer):
    USER = serializers.CharField()
    FILE = serializers.FileField()
    CONFIG = JSONObjectField(required=False)
    KEYS = JSONObjectField(required=False)

    def validate_FILE(self, value):
        from pathlib import Path
        if Path(value.name).suffix.lower() not in {".pdf", ".txt", ".md"}:
            raise serializers.ValidationError("Upload a PDF, TXT, or Markdown file.")
        if value.size > 20 * 1024 * 1024:
            raise serializers.ValidationError("Files must be 20 MB or smaller.")
        return value


class DeepAnalysisSerializer(serializers.Serializer):
    USER = serializers.CharField()
    QUERY = serializers.CharField()


class InsertTextSerializer(serializers.Serializer):
    USER = serializers.CharField()
    TEXT = serializers.CharField()
    CONFIG = JSONObjectField(required=False)
    KEYS = JSONObjectField(required=False)


class SignUpSerializer(serializers.Serializer):
    EMAIL = serializers.EmailField()
    USERNAME = serializers.CharField()
