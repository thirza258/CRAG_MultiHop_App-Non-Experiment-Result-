from django.db import models
import uuid
# Create your models here.e

class GuestUser(models.Model):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.username

class Document(models.Model):
    user = models.ForeignKey(GuestUser, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    source_type = models.CharField(
        max_length=20,
        choices=[("pdf", "PDF"), ("url", "URL"), ("text", "Text")]
    )
    file = models.TextField(null=True, blank=True)
    extracted_text_path = models.TextField(null=True, blank=True)
    source_path = models.TextField(null=True, blank=True)
    source_url = models.URLField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status = models.CharField(
    max_length=20,
    choices=[
            ("pending",  "Pending"),
            ("indexing", "Indexing"),
            ("ready",    "Ready"),
            ("failed",   "Failed"),
        ],
        default="pending", blank=True, null=True
    )
    file_hash = models.CharField(max_length=64, blank=True, null=True)
    
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["user", "file_hash"],
                name="unique_user_file_hash"
            )
        ]

    def __str__(self):
        return f"{self.name} - {self.source_type}"


class ChromaCollection(models.Model):
    """Shared corpus collection — not per user."""
    TYPE_CHOICES = [
        ("corpus", "Corpus"),
    ]
    collection_name = models.CharField(max_length=255, unique=True)
    collection_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default="corpus")
    # The model that actually produced this collection's vectors. Blank means
    # "not recorded" — query-time dense retrieval pins itself to this value, so
    # a wrong one is worse than none, and the old default asserted a specific
    # model on every new row without anything ever writing the real one.
    embedding_model = models.CharField(max_length=100, blank=True, default="")
    chunk_count = models.IntegerField(default=0)
    fingerprint = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.collection_type} - {self.collection_name}"


class UserCollection(models.Model):
    """
    One collection per user.
    Many documents can live inside it — tracked via DocumentChunk.
    """
    user = models.OneToOneField(GuestUser, on_delete=models.CASCADE, related_name="collection")
    collection_name = models.CharField(max_length=255, unique=True)
    # Set by AppRAGPipeline._build_index to the model that embedded the chunks,
    # and cleared when the last document is deleted. Blank means the collection
    # is not pinned to a vector space yet, so the user's choice still applies.
    embedding_model = models.CharField(max_length=100, blank=True, default="")
    chunk_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.collection_name}"


class DocumentChunk(models.Model):
    """
    Tracks which chunks belong to which document inside a UserCollection.
    This is what enables per-document deletion from Chroma.
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="document_chunks")
    user_collection = models.ForeignKey(UserCollection, on_delete=models.CASCADE, related_name="chunks")
    chroma_id = models.CharField(max_length=255, unique=True)  # the actual ID inside ChromaDB
    chunk_index = models.IntegerField()                         # position in original document
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("document", "chunk_index")

    def __str__(self):
        return f"{self.document.name} - chunk {self.chunk_index}"
    
class VectorStore(models.Model):
    base_path = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.base_path or "VectorStore"

class DocumentVector(models.Model):
    METHOD_CHOICES = [
        ("dense", "Dense"),
        ("sparse", "Sparse"),
        ("hybrid", "Hybrid"),
    ]
    
    document = models.ForeignKey(
        Document,
        on_delete=models.CASCADE,
        related_name="vectors"
    )
    vectorstore = models.ForeignKey(VectorStore, on_delete=models.CASCADE)
    vectorstore_location = models.TextField()
    document_location = models.TextField()
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("processing", "Processing"),
            ("ready", "Ready"),
            ("failed", "Failed"),
        ],
        default="pending"
    )
    method = models.CharField(max_length=20, choices=METHOD_CHOICES, default="dense") 
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    

    
    def __str__(self):        
        return f"Task {self.task_id} for Conversation {self.conversation_id}"
        

class Conversation(models.Model):
    user = models.ForeignKey(GuestUser, on_delete=models.CASCADE, default=None, null=True, blank=True)
    query = models.TextField()
    response = models.TextField(null=True, blank=True)
    context = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
class ConversationTask(models.Model):
    task_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(GuestUser, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class ConversationHistory(models.Model):
    user = models.ForeignKey(GuestUser, on_delete=models.CASCADE)
    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="histories"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.conversation_id)

class Job(models.Model):
    class Status(models.TextChoices):
        PENDING = "PENDING"
        PROCESSING = "PROCESSING"
        READY = "READY"
        FAILED = "FAILED"
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    progress = models.PositiveSmallIntegerField(default=0)
    user = models.ForeignKey(GuestUser, on_delete=models.CASCADE)
    document = models.ForeignKey(Document, on_delete=models.CASCADE, null=True, blank=True) 
    vectorstore = models.ForeignKey(VectorStore, on_delete=models.CASCADE, null=True, blank=True)
    error_message = models.TextField(blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.id} - {self.status}"

    def mark_failed(self, message: str):
        self.status = self.Status.FAILED
        self.error_message = message
        self.save(update_fields=["status", "error_message", "updated_at"])

class Metadata(models.Model):
    llm_model = models.CharField(max_length=100, default="gpt-4o", help_text="LLM for answer generation")
    temperature = models.FloatField(default=0.0, help_text="LLM temperature (0.0 = deterministic)")
    embedding_model = models.CharField(max_length=100, default="text-embedding-3-small", help_text="OpenAI embedding model")
    top_k = models.IntegerField(default=5, help_text="Number of documents to retrieve")
    chunk_strategy = models.CharField(max_length=32, default="paragraph", help_text='Chunking strategy, e.g., "fixed", "paragraph", "semantic"')
    chunk_size = models.IntegerField(default=500, help_text="Max characters per chunk")
    overlap = models.IntegerField(default=50, help_text="Overlap for fixed chunking only")
    vector_store_path = models.TextField(default="./vector_stores", help_text="Path to store vector indices")
    max_retries = models.IntegerField(default=3, help_text="Max iterations for iterative search")
    reranker_model = models.CharField(max_length=200, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help_text="Reranker model name")
    reranker_top_k = models.IntegerField(default=3, help_text="Top K for reranker")
    device = models.CharField(max_length=20, default="cuda", help_text="Device to use - 'cuda' or 'cpu'")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class AnalysisBatch(models.Model):  
    user = models.ForeignKey(GuestUser, on_delete=models.CASCADE)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, default=None, null=True, blank=True)
    query = models.TextField()
    job_id = models.UUIDField(default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    total_variants = models.IntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['job_id'], name='unique_job_id')
        ]

class AnalysisResult(models.Model):
    batch = models.ForeignKey(AnalysisBatch, related_name='results', on_delete=models.CASCADE)
    method = models.CharField(max_length=100)
    query = models.TextField(default="")
    ai_model = models.CharField(max_length=100)
    answer = models.TextField()
    retrieved_chunks = models.JSONField(default=list, null=True, blank=True) 
    evaluation_metrics = models.JSONField(default=list, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']