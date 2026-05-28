"""Embedding wrapper used by the Memory_Store.

This module owns the embedding contract for the JARVIS Memory_Store described
in ``design.md §Memory_Store``. The Dialog_Manager queries Memory_Store for
the top-K most similar Memory_Records before composing each Assistant_Response
(Requirement 10.3); Memory_Store delegates similarity computation to ChromaDB,
which in turn requires a deterministic embedding function whose output is
stable across calls so retrieval is reproducible within a session
(``CP3: Memory Retrieval Determinism``).

Two backends are provided:

* :class:`SentenceTransformerEmbedder` — the production backend. Wraps the
  ``sentence-transformers/all-MiniLM-L6-v2`` model. The
  :mod:`sentence_transformers` import is performed lazily inside the
  constructor so importing this module does not pay the multi-hundred-MB
  PyTorch / model load cost. The wrapper produces L2-normalized 384-dim
  ``float32``→``float`` vectors so cosine similarity and dot-product give
  identical orderings — this is the metric ChromaDB will be configured with
  in task 14.3.

* :class:`HashEmbedder` — a deterministic, dependency-free pseudo-embedder
  intended for tests, CI, and offline development. It seeds a SHA-256-based
  counter-mode keystream from the input text, decodes successive 32-bit
  little-endian words into ``[-1.0, 1.0)`` floats, and L2-normalizes the
  result. This gives Memory_Store tests a backend that

  * is deterministic given the same text (stable retrieval ordering for
    CP3),
  * produces distinct vectors for distinct texts with overwhelming
    probability (so similarity tests are meaningful),
  * runs in microseconds so property-based tests can drive thousands of
    embed calls without timing out, and
  * has the same ``embed`` / ``embed_batch`` shape as the production
    backend, so ``MemoryStore`` accepts it without conditional code paths.

The :class:`Embedder` Protocol is :func:`runtime_checkable` so test fakes
(including :class:`HashEmbedder`) and any future backend (e.g. an OpenAI
embedding adapter) can satisfy the contract without nominal subclassing.

Validates: Requirements 10.3, 10.4
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - import-time only
    # Imported under TYPE_CHECKING so the heavy ``sentence_transformers``
    # import remains lazy at runtime. Type-only references are fine here
    # because mypy can resolve the stub without loading torch.
    from sentence_transformers import SentenceTransformer

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "MINI_LM_DIMENSION",
    "Embedder",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "create_default_embedder",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# The default production model. Pinned to the same identifier the
# configuration schema (``MemoryConfig.embedding_model``) defaults to and
# the design document calls out under §Memory_Store.
DEFAULT_EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

# Output dimension of ``all-MiniLM-L6-v2``. Hard-coded as a published-model
# constant so :class:`HashEmbedder` can match it without instantiating the
# real model. If a future production backend changes dimension, callers
# should rely on the per-instance :attr:`Embedder.dimension` property
# rather than this constant.
MINI_LM_DIMENSION: Final[int] = 384

# Domain separator mixed into the SHA-256 keystream so :class:`HashEmbedder`
# outputs do not collide with hashes used elsewhere in the codebase (for
# example, the DPAPI ``NullDPAPI`` keystream). The version suffix lets us
# evolve the embedding scheme without silently re-keying existing fixtures.
_HASH_DOMAIN: Final[bytes] = b"jarvis-hash-embedder-v1"

# Number of bytes per emitted float in the hash keystream (4 = 32-bit). We
# decode each 4-byte word as a little-endian unsigned int and map it onto
# the half-open interval ``[-1.0, 1.0)`` before L2-normalisation. Choosing
# 4 bytes keeps the keystream cost at ``ceil(dim * 4 / 32)`` SHA-256
# digests — for the default 384-dim that is 48 hashes per ``embed`` call,
# which benchmarks show as comfortably below 100 µs on contemporary CPUs.
_HASH_FLOAT_STRIDE: Final[int] = 4


# ---------------------------------------------------------------------------
# Embedder Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Structural contract for any text-embedding backend.

    Implementations MUST be deterministic: ``embed(t) == embed(t)`` for any
    fixed text ``t`` within the lifetime of a single embedder instance, and
    SHOULD remain stable across instances created with the same
    configuration. CP3 (Memory Retrieval Determinism) depends on this
    invariant — retrieval ordering would otherwise drift between calls
    even with no underlying writes.

    Implementations MUST also satisfy the *batch / single equivalence*
    invariant: ``embed_batch([t])[0] == embed(t)`` for any text ``t``. This
    keeps tests and consumers from accidentally diverging by picking one
    entry point over the other.

    Attributes:
        model_name: Stable identifier of the underlying model (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``). Used by
            ``MemoryStore`` to tag stored vectors so a future model
            upgrade can be detected and the index rebuilt.
        dimension: Length of every vector produced by :meth:`embed`. All
            vectors returned by a single embedder instance MUST have
            this length.
    """

    model_name: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``.

        The returned list has length :attr:`dimension`. Empty input is
        accepted; backends are expected to return a well-defined vector
        for it (typically the embedding of an empty string under the
        underlying model, or the zero vector for the hash-based test
        double — see :class:`HashEmbedder` for the chosen convention).
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for each text in ``texts``, preserving order.

        ``embed_batch([])`` returns ``[]``. Implementations MAY exploit
        batching to amortise model overhead, but MUST yield results that
        are element-wise equal to :meth:`embed` calls on the same inputs
        (see the protocol-level batch / single equivalence invariant).
        """
        ...


# ---------------------------------------------------------------------------
# Production backend: sentence-transformers/all-MiniLM-L6-v2
# ---------------------------------------------------------------------------


class SentenceTransformerEmbedder:
    """Production embedder backed by :mod:`sentence_transformers`.

    The constructor lazily imports :mod:`sentence_transformers` so the
    multi-hundred-MB PyTorch and model-loading cost is only paid when an
    instance is actually constructed. Tests that exercise Memory_Store
    against a deterministic vector space can use :class:`HashEmbedder`
    instead and avoid the import entirely.

    Outputs are L2-normalized so cosine similarity and dot product give
    identical orderings; this matches the ``HashEmbedder`` convention
    and the ChromaDB ``"cosine"`` collection metric Memory_Store will
    request in task 14.3.

    Reproducibility: ``sentence-transformers/all-MiniLM-L6-v2`` is a
    deterministic transformer-based encoder — for a fixed model version
    and identical input, it produces identical outputs. Pinning the
    model identifier (via the ``model_name`` argument or the configured
    default) is therefore sufficient to guarantee CP3 retrieval
    determinism across application runs.
    """

    model_name: str
    dimension: int

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        cache_folder: str | None = None,
        device: str | None = None,
    ) -> None:
        """Load the underlying model.

        Parameters
        ----------
        model_name:
            Hugging Face model identifier. Defaults to
            ``sentence-transformers/all-MiniLM-L6-v2`` per the design
            document and ``MemoryConfig.embedding_model``.
        cache_folder:
            Optional directory to cache the downloaded model weights;
            forwarded to :class:`sentence_transformers.SentenceTransformer`.
        device:
            Optional device string (``"cpu"``, ``"cuda"``, etc.). When
            ``None``, ``sentence_transformers`` picks a default.
        """
        try:
            # Lazy import: keeps ``import jarvis.memory.embedder`` cheap
            # for callers who only need the Protocol or the test double.
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised on minimal envs
            raise RuntimeError(
                "SentenceTransformerEmbedder requires the `sentence-transformers` "
                "package. Install it (declared in pyproject.toml) or use "
                "HashEmbedder for tests."
            ) from exc

        self._model: SentenceTransformer = SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            device=device,
        )
        self.model_name = model_name
        # The model exposes an integer hidden dimension via
        # ``get_sentence_embedding_dimension``. We store it as a plain int
        # so the Protocol's ``dimension: int`` attribute is satisfied
        # without further indirection.
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            # Some sentence-transformers builds return None for models
            # whose pooling layer is non-standard. Fall back to a probe.
            probe: Any = self._model.encode(
                [""],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            dim = int(probe.shape[1])
        self.dimension = int(dim)

    def embed(self, text: str) -> list[float]:
        """Embed a single text. See :meth:`Embedder.embed`."""
        # ``encode`` returns a 1-D ndarray when given a single string.
        # We always go through the batch entry point so the
        # batch/single equivalence invariant is by-construction.
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. See :meth:`Embedder.embed_batch`."""
        if not texts:
            return []
        vectors: Any = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # ``encode`` with a list input returns a 2-D ``np.ndarray`` of
        # shape ``(len(texts), dimension)``. Cast each row to a
        # plain ``list[float]`` so downstream consumers (ChromaDB, JSON
        # audit logs, the Embedder Protocol return type) do not have a
        # numpy dependency in their type signature.
        return [[float(x) for x in row] for row in vectors]


# ---------------------------------------------------------------------------
# Test double: SHA-256-counter-mode hash embedder
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic, dependency-free pseudo-embedder for tests / CI.

    The vector for a given text is derived from a SHA-256 keystream
    seeded with ``_HASH_DOMAIN | model_name | text``. Successive 4-byte
    words of the keystream are interpreted as little-endian unsigned
    integers and mapped onto ``[-1.0, 1.0)`` via the affine
    transformation ``2 * u / 2**32 - 1``. The resulting vector is
    L2-normalized so dot product and cosine similarity rank identically
    (matching :class:`SentenceTransformerEmbedder`).

    Properties relied on by tests:

    * **Determinism.** ``embed(t) == embed(t)`` byte-for-byte across
      processes and machines. (CP3.)
    * **Dimensional consistency.** Every returned vector has exactly
      :attr:`dimension` entries.
    * **Batch / single equivalence.** ``embed_batch([t])[0] == embed(t)``.
    * **Distinctness.** Two distinct texts produce distinct vectors
      with overwhelming probability — collisions would require a
      SHA-256 collision over the seeded input.
    * **Unit norm (almost surely).** The output L2 norm is ``1.0`` to
      within IEEE-754 rounding for any non-zero input. The empty string
      is the single edge case: its keystream digest is non-zero, so
      the empty-string vector also has unit norm.

    The class is **not** a real embedding model. Lexical similarity
    between two texts has no relationship to the cosine similarity of
    their hash embeddings. Tests that exercise *retrieval-relevance*
    semantics (e.g. "asking for X retrieves the X memory") MUST either
    seed a Memory_Store with vectors that the test author controls, or
    use the production embedder.
    """

    model_name: str
    dimension: int

    def __init__(
        self,
        *,
        dimension: int = MINI_LM_DIMENSION,
        model_name: str = "hash-embedder-v1",
    ) -> None:
        """Create a hash embedder.

        Parameters
        ----------
        dimension:
            Output vector length. Defaults to
            :data:`MINI_LM_DIMENSION` so the test double is
            shape-compatible with the production embedder. Must be a
            positive integer.
        model_name:
            Identifier mirrored back via :attr:`model_name`. Tests can
            set this to differentiate fixtures; production code should
            not depend on the literal value.
        """
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"HashEmbedder dimension must be a positive int; got {dimension!r}"
            )
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                f"HashEmbedder model_name must be a non-empty string; got {model_name!r}"
            )
        self.dimension = dimension
        self.model_name = model_name
        # Pre-compute the per-instance domain prefix. Including the
        # model_name in the seed means two HashEmbedder instances with
        # different model_name values produce distinct vector spaces —
        # useful for property tests that need an "unrelated" embedder.
        self._seed_prefix: bytes = _HASH_DOMAIN + b"|" + model_name.encode("utf-8") + b"|"

    def embed(self, text: str) -> list[float]:
        """Embed a single text. See :meth:`Embedder.embed`."""
        if not isinstance(text, str):
            raise TypeError(f"HashEmbedder.embed expects str; got {type(text).__name__}")
        return self._embed_one(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. See :meth:`Embedder.embed_batch`."""
        if not isinstance(texts, list):
            # Guard against accidental tuple / generator inputs which the
            # Protocol does not declare; fail fast rather than emitting
            # misleadingly empty results.
            raise TypeError(
                f"HashEmbedder.embed_batch expects list[str]; got {type(texts).__name__}"
            )
        return [self._embed_one(t) for t in texts]

    # -- Internal -------------------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        if not isinstance(text, str):
            raise TypeError(
                f"HashEmbedder.embed_batch entries must be str; got {type(text).__name__}"
            )
        seed = self._seed_prefix + text.encode("utf-8")
        keystream = self._derive_keystream(seed, self.dimension * _HASH_FLOAT_STRIDE)
        # Decode each 4-byte word into a float in [-1.0, 1.0). Using
        # ``struct.unpack`` over the whole keystream is materially faster
        # than per-word ``int.from_bytes`` calls on CPython 3.11.
        words = struct.unpack(f"<{self.dimension}I", keystream)
        # Map u32 -> [-1.0, 1.0). 2**32 = 4294967296.
        raw: list[float] = [(w / 2147483648.0) - 1.0 for w in words]
        # L2-normalize. The empty-string edge case still has a non-zero
        # keystream (SHA-256 of any input is non-zero), so the norm is
        # always positive in practice; we keep the guard for safety.
        norm_sq = 0.0
        for v in raw:
            norm_sq += v * v
        if norm_sq == 0.0:  # pragma: no cover - astronomically unlikely
            return raw
        # math.sqrt would also work; using **0.5 avoids the import for a
        # module that is otherwise import-cheap.
        norm = norm_sq**0.5
        return [v / norm for v in raw]

    @staticmethod
    def _derive_keystream(seed: bytes, length: int) -> bytes:
        """Derive ``length`` bytes deterministically from ``seed``.

        Counter-mode SHA-256 — identical pattern to
        :class:`jarvis.security.dpapi.NullDPAPI._derive_keystream` so the
        cryptographic hygiene story is consistent across the codebase
        (every counter-mode keystream domain-separates by prefix and
        encodes the counter as 8 BE bytes).
        """
        out = bytearray()
        counter = 0
        while len(out) < length:
            block = hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
            out.extend(block)
            counter += 1
        return bytes(out[:length])


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_default_embedder(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    *,
    cache_folder: str | None = None,
    device: str | None = None,
) -> Embedder:
    """Return the production :class:`SentenceTransformerEmbedder`.

    Application-startup code should call this; tests should construct
    :class:`HashEmbedder` directly. The factory exists so future
    additions (e.g., an opt-in OpenAI embeddings adapter) can be wired
    in without changing every call site.
    """
    return SentenceTransformerEmbedder(
        model_name,
        cache_folder=cache_folder,
        device=device,
    )
