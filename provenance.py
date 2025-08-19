from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import hashlib
import threading
import re

@dataclass
class CitationRecord:
    doc_id: str
    source: str
    page_numbers: Optional[List[int]] = None
    file_path: Optional[str] = None
    snippet_hash: Optional[str] = None
    score: Optional[float] = None
    used_in_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProvenanceStore:
    """Central store tracking all document snippets / sources used across CoT stages."""
    def __init__(self):
        self._records: Dict[str, CitationRecord] = {}
        self._order: List[str] = []  # preserve first appearance order
        self._lock = threading.Lock()
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"D{self._counter}"  # Deterministic, stable ordering by first use

    def _hash_snippet(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def add_source(self, item: Dict[str, Any], step: Optional[str] = None, score: Optional[float] = None) -> str:
        """Register a context/retrieved item.
        item expected keys: content, metadata{source, page_numbers?, file_path?}
        Returns doc_id."""
        if not item:
            return ""
        metadata = item.get("metadata", {})
        source = metadata.get("source") or metadata.get("file_path") or "Unknown"
        snippet_hash = self._hash_snippet(item.get("content", "")[:500])  # hash first 500 chars for stability

        # Build a natural key to avoid duplicates: source+pages+hash
        pages = metadata.get("page_numbers") or metadata.get("pages")
        natural_key = f"{source}|{pages}|{snippet_hash}"

        with self._lock:
            # Reuse existing record if natural key matches
            for rec in self._records.values():
                if rec.metadata.get("_natural_key") == natural_key:
                    if step and step not in rec.used_in_steps:
                        rec.used_in_steps.append(step)
                    return rec.doc_id

            doc_id = self._next_id()
            record = CitationRecord(
                doc_id=doc_id,
                source=source,
                page_numbers=pages if isinstance(pages, list) else None,
                file_path=metadata.get("file_path"),
                snippet_hash=snippet_hash,
                score=score,
                used_in_steps=[step] if step else [],
                metadata={**metadata, "_natural_key": natural_key}
            )
            self._records[doc_id] = record
            self._order.append(doc_id)
            return doc_id

    def mark_used(self, doc_id: str, step: str):
        if not doc_id or doc_id not in self._records:
            return
        with self._lock:
            rec = self._records[doc_id]
            if step not in rec.used_in_steps:
                rec.used_in_steps.append(step)

    def list_records(self) -> List[CitationRecord]:
        return [self._records[i] for i in self._order]

    # --- New: marker / citation utilities ---
    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    @classmethod
    def extract_marker_spans(cls, text: str) -> List[Dict[str, Any]]:
        """Return list of spans: {marker:int,start:int,end:int} for [#] markers in text."""
        spans: List[Dict[str, Any]] = []
        if not text:
            return spans
        for m in cls.CITATION_PATTERN.finditer(text):
            try:
                num = int(m.group(1))
            except ValueError:
                continue
            spans.append({"marker": num, "start": m.start(), "end": m.end()})
        return spans

    @staticmethod
    def map_markers_to_doc_ids(spans: List[Dict[str, Any]], marker_map: Dict[int, List[str]]) -> List[Dict[str, Any]]:
        """Augment spans with doc_ids from marker_map."""
        enriched = []
        for s in spans:
            doc_ids = marker_map.get(s["marker"], []) if marker_map else []
            enriched.append({**s, "doc_ids": doc_ids})
        return enriched

    @staticmethod
    def validate_marker_mapping(spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that each span has at least one doc_id. Return summary."""
        total = len(spans)
        orphan = [s for s in spans if not s.get("doc_ids")]
        return {
            "total_markers": total,
            "orphan_markers": len(orphan),
            "orphan_details": orphan[:20],  # cap for brevity
            "all_valid": len(orphan) == 0
        }

    def to_reference_list(self) -> List[Dict[str, Any]]:
        """Return a serializable list for API / UI consumption."""
        refs = []
        for rec in self.list_records():
            refs.append({
                "id": rec.doc_id,
                "source": rec.source,
                "pages": rec.page_numbers,
                "file_path": rec.file_path,
                "score": rec.score,
                "used_in_steps": rec.used_in_steps,
                "metadata": {k: v for k, v in rec.metadata.items() if k != "_natural_key"}
            })
        return refs

    # Convenience: full validation pipeline for a text + marker map
    def validate_text_with_map(self, text: str, marker_map: Dict[int, List[str]]) -> Dict[str, Any]:
        spans = self.extract_marker_spans(text)
        enriched = self.map_markers_to_doc_ids(spans, marker_map)
        summary = self.validate_marker_mapping(enriched)
        summary["spans"] = enriched
        return summary

    def clear(self):
        with self._lock:
            self._records.clear()
            self._order.clear()
            self._counter = 0
