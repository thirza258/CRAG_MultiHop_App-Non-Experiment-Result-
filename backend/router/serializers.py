from rest_framework import serializers

class QuerySerializer(serializers.Serializer):
    USER = serializers.CharField()
    QUERY = serializers.CharField()

class InsertURLSerializer(serializers.Serializer):
    USER = serializers.CharField()
    URL = serializers.URLField()

class InsertDataSerializer(serializers.Serializer):
    USER = serializers.CharField()
    FILE = serializers.FileField()

class DeepAnalysisSerializer(serializers.Serializer):
    USER = serializers.CharField()
    QUERY = serializers.CharField()

class InsertTextSerializer(serializers.Serializer):
    USER = serializers.CharField()
    TEXT = serializers.CharField()

class SignUpSerializer(serializers.Serializer):
    EMAIL = serializers.EmailField()
    USERNAME = serializers.CharField()
