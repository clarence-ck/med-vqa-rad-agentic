#!/usr/bin/env python3
"""
RAG Pipeline Test Script
Tests the Advanced RAG retrieval system with various scenarios.
"""

import sys
import time
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from agents.advanced_rag import (
    AdvancedRAGRetriever,
    retrieve_knowledge_advanced,
    get_rag_retriever,
)
from agents.knowledge_retriever import KnowledgeRetriever


def test_direct_rag_retrieval():
    """Test the low-level RAG retriever with various queries."""
    print("\n" + "="*60)
    print("TEST 1: Direct RAG Retrieval")
    print("="*60)

    test_cases = [
        {
            "name": "Chest X-ray - Pneumonia",
            "question": "Is there evidence of pneumonia in the right lower lobe?",
            "modality": "x-ray",
            "body_region": "chest",
            "finding_category": "pneumonia",
        },
        {
            "name": "CT Head - Hemorrhage",
            "question": "What type of intracranial hemorrhage is present?",
            "modality": "ct",
            "body_region": "head",
            "finding_category": "hemorrhage",
        },
        {
            "name": "MRI Brain - Mass",
            "question": "Describe the mass effect and surrounding edema.",
            "modality": "mri",
            "body_region": "head",
            "finding_category": "mass",
        },
        {
            "name": "X-ray - Fracture",
            "question": "Is there a fracture in the distal radius?",
            "modality": "x-ray",
            "body_region": "musculoskeletal",
            "finding_category": "fracture",
        },
        {
            "name": "No metadata filter",
            "question": "What abnormality is seen in this image?",
            "modality": None,
            "body_region": None,
            "finding_category": None,
        },
    ]

    retriever = get_rag_retriever()
    results_summary = []

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print(f"Query: {tc['question'][:60]}...")
        print(f"Filters: modality={tc['modality']}, region={tc['body_region']}, finding={tc['finding_category']}")

        start = time.perf_counter()
        result = retriever.retrieve(
            question=tc["question"],
            modality=tc["modality"],
            body_region=tc["body_region"],
            finding_category=tc["finding_category"],
        )
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Retrieved: {len(result)} cases in {elapsed:.1f}ms")

        if result:
            for i, r in enumerate(result[:3], 1):
                print(f"  {i}. Score={r.score:.3f} | {r.metadata.get('modality', '?')}/{r.metadata.get('body_region', '?')}")
                print(f"     Q: {r.question[:70]}...")
                print(f"     A: {r.answer[:70]}...")
        else:
            print("  ⚠️  No cases retrieved (low confidence or no matches)")

        results_summary.append({
            "name": tc["name"],
            "count": len(result),
            "duration_ms": elapsed,
            "top_score": result[0].score if result else 0.0,
        })

    return results_summary


def test_metadata_filtering():
    """Test that metadata filtering actually narrows search space."""
    print("\n" + "="*60)
    print("TEST 2: Metadata Filtering Effectiveness")
    print("="*60)

    retriever = get_rag_retriever()
    query = "Is there a mass lesion?"

    scenarios = [
        ("No filter", None, None, None),
        ("Modality only: MRI", "mri", None, None),
        ("Region only: head", None, "head", None),
        ("Both: MRI head", "mri", "head", None),
        ("Full: MRI head mass", "mri", "head", "mass"),
    ]

    for name, mod, region, finding in scenarios:
        candidates = retriever._apply_metadata_filter(mod, region, finding)
        count = len(candidates) if candidates else "full corpus"
        print(f"  {name:25} → {count} candidates")


def test_finding_category_inference():
    """Test finding category inference from question + notable findings."""
    print("\n" + "="*60)
    print("TEST 3: Finding Category Inference")
    print("="*60)

    retriever = get_rag_retriever()

    test_inputs = [
        ("fracture of the femur", "fracture"),
        ("intracranial hemorrhage", "hemorrhage"),
        ("large mass with edema", "mass"),
        ("pleural effusion present", "effusion"),
        ("consolidation in right lung", "pneumonia"),
        ("normal chest x-ray", "normal"),
        ("chest tube in place", "device"),
    ]

    for text, expected in test_inputs:
        result = retriever._infer_finding_category("What is the finding?", text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{text[:40]}...' → {result} (expected: {expected})")


def test_retrieval_confidence():
    """Test confidence scoring on various query difficulties."""
    print("\n" + "="*60)
    print("TEST 4: Retrieval Confidence Scoring")
    print("="*60)

    retriever = get_rag_retriever()

    queries = [
        ("Specific, well-formed", "Is there a left-sided pneumothorax on this chest x-ray?"),
        ("Vague", "What do you see?"),
        ("Non-medical", "What color is the sky?"),
        ("Abbreviation heavy", "ICH with midline shift?"),
    ]

    for name, query in queries:
        print(f"\n--- {name}: '{query[:50]}...' ---")

        # Get dense + sparse + reranked for confidence calc
        dense = retriever._dense_retrieval(query, None)
        sparse = retriever._sparse_retrieval(query, None)
        merged = retriever._merge_results(dense, sparse)
        reranked = retriever._rerank(query, merged)[:3]

        if reranked:
            confidence = retriever._compute_retrieval_confidence(dense, sparse, reranked)
            print(f"  Confidence level: {confidence['level']}")
            print(f"  Weak signals: {confidence['weak_signals']}/4")
            print(f"  Top1 softmax: {confidence['top1_softmax']:.3f}")
            print(f"  Margin: {confidence['softmax_margin']:.3f}")
            print(f"  Dense-sparse agreement: {confidence['top_candidate_in_dense_and_sparse_top5']}")
        else:
            print("  No candidates after reranking")


def test_knowledge_retriever_integration():
    """Test the full KnowledgeRetriever with RAG."""
    print("\n" + "="*60)
    print("TEST 5: KnowledgeRetriever Integration")
    print("="*60)

    kr = KnowledgeRetriever(use_advanced_rag=True)

    # Mock inputs
    image_interpretation = {
        "modality": "x-ray",
        "body_region": "chest",
        "visible_structures": ["lungs", "heart", "ribs", "diaphragm"],
        "notable_findings": ["opacity", "consolidation"],
    }

    question_analysis = {
        "question_type": "abnormality_detection",
        "target_entity": "right lower lobe",
    }

    question = "Is there pneumonia in the right lower lobe?"

    print(f"Question: {question}")
    print(f"Image: {image_interpretation['modality']} of {image_interpretation['body_region']}")

    start = time.perf_counter()
    result = kr.run(
        image_interpretation={"result": image_interpretation},
        question_analysis={"result": question_analysis},
        original_question=question,
    )
    elapsed = (time.perf_counter() - start) * 1000

    result_data = result.get("result", {})

    print(f"\nDuration: {elapsed:.1f}ms")
    print(f"Status: {result_data.get('status', 'unknown')}")
    print(f"Key facts ({len(result_data.get('key_facts', []))}):")
    for fact in result_data.get('key_facts', [])[:5]:
        print(f"  - {fact}")

    rag_info = result_data.get("rag_retrieval", {})
    print(f"\nRAG: {rag_info.get('metadata', {}).get('count', 0)} cases retrieved")
    if rag_info.get("similar_cases"):
        for i, case in enumerate(rag_info["similar_cases"][:3], 1):
            print(f"  {i}. [{case.get('score', 0):.3f}] {case.get('answer', 'N/A')[:50]}...")


def test_rag_vs_no_rag():
    """Compare retrieval with and without metadata filtering."""
    print("\n" + "="*60)
    print("TEST 6: With vs Without Metadata Filtering")
    print("="*60)

    query = "Fracture of the distal radius"

    # With metadata
    with_filter = retrieve_knowledge_advanced(
        question=query,
        modality="x-ray",
        body_region="musculoskeletal",
        finding_category="fracture",
    )

    # Without metadata
    without_filter = retrieve_knowledge_advanced(
        question=query,
        modality=None,
        body_region=None,
        finding_category=None,
    )

    print(f"Query: {query}")
    print(f"\nWith metadata filter (x-ray, musculoskeletal, fracture):")
    print(f"  Retrieved: {with_filter['count']} cases")
    for c in with_filter['similar_cases'][:2]:
        print(f"    - {c['metadata']} | {c['answer'][:50]}...")

    print(f"\nWithout metadata filter:")
    print(f"  Retrieved: {without_filter['count']} cases")
    for c in without_filter['similar_cases'][:2]:
        print(f"    - {c['metadata']} | {c['answer'][:50]}...")


def main():
    print("RAG Pipeline Test Suite")
    print("="*60)

    tests = [
        ("Direct RAG Retrieval", test_direct_rag_retrieval),
        ("Metadata Filtering", test_metadata_filtering),
        ("Finding Category Inference", test_finding_category_inference),
        ("Retrieval Confidence", test_retrieval_confidence),
        ("KnowledgeRetriever Integration", test_knowledge_retriever_integration),
        ("With vs Without Metadata", test_rag_vs_no_rag),
    ]

    results = {}

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                results[name] = result
            print(f"\n✅ {name} PASSED")
        except Exception as e:
            print(f"\n❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if "Direct RAG Retrieval" in results:
        print("\nRetrieval Performance:")
        for r in results["Direct RAG Retrieval"]:
            print(f"  {r['name'][:30]:30} | Count: {r['count']} | Top Score: {r['top_score']:.3f} | {r['duration_ms']:.0f}ms")

    print("\nAll tests completed!")


if __name__ == "__main__":
    main()
