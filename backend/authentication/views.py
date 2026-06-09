from django.shortcuts import render
from rest_framework.generics import GenericAPIView
from router.serializers import SignUpSerializer
from router.models import GuestUser
from common.schema import get_responses
from chroma.chroma_settings import create_chroma_collection

# Create your views here.
class SignUpView(GenericAPIView):
    serializer_class = SignUpSerializer

    def post(self, request):
        try:
            serializer = self.get_serializer(data=request.data)
            serializer.is_valid(raise_exception=True)

            email = serializer.validated_data["EMAIL"]
            username = serializer.validated_data["USERNAME"]

            user, created = GuestUser.objects.get_or_create(
                email=email,
                username=username,
            )
            
            response_dict = {
                "response": "User created successfully!" if created else "User already exists",
                "username": username,
                "email": email,
            }
            if created:
                return get_responses().response_201(response_dict)
            else:
                return get_responses().response_200(response_dict)
        except Exception as e:
            return get_responses().response_500(error=str(e))