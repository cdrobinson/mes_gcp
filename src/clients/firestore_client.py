"""Firestore client."""

from .base_client import BaseClient
from google.cloud import firestore

class FirestoreClient(BaseClient):
    """Firestore client for interacting with Google Cloud Firestore."""

    def __init__(self, project_id: str):
        """Initialise the Firestore client."""
        super().__init__(project_id)
        self.db = firestore.Client(project=self.project_id)

    def get_document(self, collection_id: str, document_id: str) -> dict:
        """
        Gets a document from a Firestore collection.

        Args:
            collection_id: The ID of the collection.
            document_id: The ID of the document.

        Returns:
            The document data.
        """
        doc_ref = self.db.collection(collection_id).document(document_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        else:
            raise FileNotFoundError(f"Document {document_id} not found in collection {collection_id}")

    def query_collection(self, collection_id: str, query_params: list) -> list:
        """
        Queries a Firestore collection.

        Args:
            collection_id: The ID of the collection.
            query_params: A list of tuples representing the query parameters.
                          e.g. [("field", "==", "value")]

        Returns:
            A list of documents that match the query.
        """
        collection_ref = self.db.collection(collection_id)
        query = collection_ref
        for param in query_params:
            query = query.where(*param)
        
        docs = query.stream()
        return [doc.to_dict() for doc in docs]
