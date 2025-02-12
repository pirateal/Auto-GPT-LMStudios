import dataclasses
import orjson
from typing import Any, List, Optional
import numpy as np
import os
from memory.base import MemoryProviderSingleton, get_ada_embedding

EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS

def create_default_embeddings():
    return np.zeros((0, EMBED_DIM)).astype(np.float32)

@dataclasses.dataclass
class CacheContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings
    )

class LocalCache(MemoryProviderSingleton):
    # on load, load our database
    def __init__(self, cfg) -> None:
        self.filename = f"{cfg.memory_index}.json"
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'w+b') as f:
                    file_content = f.read()
                    if not file_content.strip():
                        file_content = b'{}'
                        f.write(file_content)

                    loaded = orjson.loads(file_content)
                    self.data = CacheContent(**loaded)
            except orjson.JSONDecodeError:
                print(f"Error: The file '{self.filename}' is not in JSON format.")
                self.data = CacheContent()
        else:
            print(f"Warning: The file '{self.filename}' does not exist. Local memory would not be saved to a file.")
            self.data = CacheContent()

    def add(self, text: str):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        if 'Command Error:' in text:
            return ""
        self.data.texts.append(text)

        embedding = get_ada_embedding(text)
        vector = np.array(embedding).astype(np.float32)

        # Ensure vector is two-dimensional
        if vector.ndim == 3:
            # Assuming we need to take the mean across the sequence length dimension
            vector = vector.mean(axis=1)

        # Ensure vector is reshaped as a single row for concatenation
        vector = vector.reshape(1, -1)

        self.data.embeddings = np.concatenate(
            [
                self.data.embeddings,
                vector,
            ],
            axis=0,
        )

        with open(self.filename, 'wb') as f:
            out = orjson.dumps(
                self.data,
                option=SAVE_OPTIONS
            )
            f.write(out)
        return text


    def clear(self) -> str:
        """
        Clears the redis server.

        Returns: A message indicating that the memory has been cleared.
        """
        self.data = CacheContent()
        return "Obliviated"

    def get(self, data: str) -> Optional[List[Any]]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def get_relevant(self, text: str, k: int) -> List[Any]:
        """
        matrix-vector mult to find score-for-each-row-of-matrix
        get indices for top-k winning scores
        return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        embedding = get_ada_embedding(text)
        
        # Reshape the embedding for dot product
        embedding = embedding.reshape(-1, 1)  # Reshaping to (1536, 1)

        scores = np.dot(self.data.embeddings, embedding)

        top_k_indices = np.argsort(scores.flatten())[-k:][::-1]

        # Convert top_k_indices to a list of integers
        top_k_indices = top_k_indices.tolist()

        return [self.data.texts[i] for i in top_k_indices]




    def get_stats(self):
        """
        Returns: The stats of the local cache.
        """
        return len(self.data.texts), self.data.embeddings.shape
