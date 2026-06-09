import uuid

def _document_base_path(username: str) -> str:
    doc_id = uuid.uuid4().hex
    return f"documents/user_{username}/{doc_id}"

def conversation_id_generator() -> str:
    return uuid.uuid4().hex