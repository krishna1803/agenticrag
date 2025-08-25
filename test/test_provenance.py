import pytest
import sys
import os

# Add the parent directory (project root) to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from provenance import ProvenanceStore

def test_add_source_deduplication():
    store = ProvenanceStore()
    item1 = {"content": "Sample text A", "metadata": {"source": "docA.pdf", "page_numbers": [1]}}
    item2 = {"content": "Sample text A", "metadata": {"source": "docA.pdf", "page_numbers": [1]}}
    id1 = store.add_source(item1, step="research")
    id2 = store.add_source(item2, step="reason")
    assert id1 == id2
    recs = store.to_reference_list()
    assert recs[0]["used_in_steps"] == ["research", "reason"] or recs[0]["used_in_steps"] == ["research", "reason"]


def test_extract_marker_spans():
    spans = ProvenanceStore.extract_marker_spans("Fact [1] and [2] plus [10]")
    markers = [s["marker"] for s in spans]
    assert markers == [1,2,10]
    assert spans[0]["start"] < spans[0]["end"]


def test_map_and_validate():
    spans = ProvenanceStore.extract_marker_spans("Text [1] x [3] y [3]")
    marker_map = {1:["D1"],3:["D2","D3"]}
    enriched = ProvenanceStore.map_markers_to_doc_ids(spans, marker_map)
    assert enriched[0]["doc_ids"] == ["D1"]
    summary = ProvenanceStore.validate_marker_mapping(enriched)
    assert summary["all_valid"]


def test_validate_text_with_map_orphans():
    store = ProvenanceStore()
    res = store.validate_text_with_map("Some [4] orphan", {1:["D1"]})
    assert res["total_markers"] == 1
    assert res["orphan_markers"] == 1
    assert not res["all_valid"]
