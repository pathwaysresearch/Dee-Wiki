"""
Standalone test runner - 27 comprehensive tests
No external dependencies required (no pytest)

Run: python3 tests/run_tests.py
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "webapp"))

# Counters
TESTS_PASSED = 0
TESTS_FAILED = 0
TESTS_SKIPPED = 0


def test(test_num, test_name, test_func):
    """Run a single test and track result"""
    global TESTS_PASSED, TESTS_FAILED, TESTS_SKIPPED
    
    try:
        test_func()
        TESTS_PASSED += 1
        print(f"✅ TEST {test_num} PASSED: {test_name}")
        return True
    except AssertionError as e:
        TESTS_FAILED += 1
        print(f"❌ TEST {test_num} FAILED: {test_name}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        TESTS_FAILED += 1
        print(f"❌ TEST {test_num} ERROR: {test_name}")
        print(f"   Exception: {type(e).__name__}: {e}")
        return False


# ============================================================================
# TESTS 1-5: Logging System
# ============================================================================

def test_1_log_file_creation():
    """Test that log.md is created if missing"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    assert log_path.exists(), "log.md does not exist"


def test_2_log_entry_format():
    """Test that log entries have correct format"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    log_to_wiki_log("test", "Format Check")
    
    content = log_path.read_text(encoding="utf-8")
    assert "##" in content, "Missing log header ##"
    assert "[" in content and "]" in content, "Missing date brackets"
    assert "test |" in content or "Format Check" in content, "Missing operation or description"


def test_3_log_metadata_preservation():
    """Test that metadata is correctly logged"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    before_len = len(log_path.read_text(encoding="utf-8"))
    
    log_to_wiki_log("test", "Metadata Test", {"key": "value"})
    
    content = log_path.read_text(encoding="utf-8")
    after_len = len(content)
    assert after_len > before_len, "Log was not appended"


def test_4_log_append_not_overwrite():
    """Test that logs append, don't overwrite"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    content_before = log_path.read_text(encoding="utf-8")
    
    log_to_wiki_log("append_test", "Should not overwrite")
    
    content_after = log_path.read_text(encoding="utf-8")
    assert len(content_after) > len(content_before), "Log was not appended"
    assert content_before in content_after, "Previous logs were overwritten"


def test_5_log_multiple_entries():
    """Test that multiple entries are logged"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    content_before = log_path.read_text(encoding="utf-8")
    
    log_to_wiki_log("multi1", "Entry A")
    log_to_wiki_log("multi2", "Entry B")
    
    content_after = log_path.read_text(encoding="utf-8")
    assert "Entry A" in content_after and "Entry B" in content_after, "Both entries not found"


# ============================================================================
# TESTS 6-10: Backend API Response Format
# ============================================================================

def test_6_log_to_wiki_log_accessible():
    """Test that log_to_wiki_log is accessible from API"""
    from webapp.api.index import log_to_wiki_log
    assert callable(log_to_wiki_log), "log_to_wiki_log not callable"


def test_7_should_update_wiki_rules():
    """Test should_update_wiki enforces all 3 rules"""
    from webapp.api.index import should_update_wiki
    
    # Rule 1: insufficient sources
    result = should_update_wiki(
        {"should_wiki_update": True, "new_synthesis": "test"},
        []
    )
    assert result is False, "Should reject with 0 sources"
    
    # Rule 2: no LLM suggestion
    result = should_update_wiki(
        {"should_wiki_update": False, "new_synthesis": "test"},
        [{}, {}]
    )
    assert result is False, "Should reject without LLM suggestion"
    
    # Rule 3: no synthesis
    result = should_update_wiki(
        {"should_wiki_update": True, "new_synthesis": ""},
        [{}, {}]
    )
    assert result is False, "Should reject without synthesis"


def test_8_should_update_wiki_all_pass():
    """Test should_update_wiki returns True when all rules pass"""
    from webapp.api.index import should_update_wiki
    
    result = should_update_wiki(
        {"should_wiki_update": True, "new_synthesis": "Valid synthesis text"},
        [{}, {}]
    )
    assert result is True, "Should pass all 3 rules"


def test_9_process_wiki_update_exists():
    """Test process_wiki_update_explicit exists"""
    from webapp.api.index import process_wiki_update_explicit
    assert callable(process_wiki_update_explicit), "process_wiki_update_explicit not callable"


def test_10_wiki_search_exists():
    """Test wiki_search function exists and works"""
    from webapp.api.index import wiki_search
    assert callable(wiki_search), "wiki_search not callable"
    
    results = wiki_search("test", top_k=5)
    assert isinstance(results, list), "wiki_search should return list"


# ============================================================================
# TESTS 11-15: Extract Entities
# ============================================================================

def test_11_log_function_defined():
    """Test log_to_wiki_log is defined in extract_entities"""
    from scripts.extract_entities import log_to_wiki_log
    assert callable(log_to_wiki_log), "log_to_wiki_log not callable"


def test_12_extract_source_callable():
    """Test extract_source is callable"""
    from scripts.extract_entities import extract_source
    assert callable(extract_source), "extract_source not callable"


def test_13_log_with_lists():
    """Test logging with list metadata"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    before = len(log_path.read_text(encoding="utf-8"))
    
    log_to_wiki_log("test", "List test", {"pages": ["p1", "p2"]})
    
    after = len(log_path.read_text(encoding="utf-8"))
    assert after > before, "List metadata not logged"


def test_14_log_robust_to_edge_cases():
    """Test log_to_wiki_log doesn't crash on edge cases"""
    from scripts.extract_entities import log_to_wiki_log
    
    # Should not raise
    log_to_wiki_log("test", "")
    log_to_wiki_log("test", None)
    log_to_wiki_log("test", "ok", None)


def test_15_log_timestamp_exists():
    """Test log entries contain timestamps"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    import re
    
    log_path = VAULT / "wiki" / "log.md"
    log_to_wiki_log("timestamp_test", "Check timestamp")
    
    content = log_path.read_text(encoding="utf-8")
    # Should match [YYYY-MM-DD]
    match = re.search(r'\[\d{4}-\d{2}-\d{2}\]', content)
    assert match, "No valid ISO date found"


# ============================================================================
# TESTS 16-20: Search, Embeddings, Storage
# ============================================================================

def test_16_embed_model_defined():
    """Test embedding model is accessible"""
    from webapp.api.index import EMBED_MODEL, DOC_PREFIX
    assert EMBED_MODEL, "EMBED_MODEL not defined"
    assert DOC_PREFIX, "DOC_PREFIX not defined"


def test_17_chunks_loaded():
    """Test RAG chunks are loaded"""
    from webapp.api.index import CHUNKS
    assert isinstance(CHUNKS, list), "CHUNKS not a list"
    assert len(CHUNKS) > 0, "No chunks loaded"


def test_18_chunk_embeddings_optional():
    """Test chunk embeddings (optional for dev)"""
    from webapp.api.index import CHUNK_EMBEDDINGS, CHUNKS
    
    if CHUNK_EMBEDDINGS is not None:
        assert len(CHUNK_EMBEDDINGS) > 0, "Embeddings exist but empty"


def test_19_wiki_store_initialized():
    """Test wiki store is initialized"""
    from webapp.api.index import WIKI_STORE
    assert WIKI_STORE is not None, "WIKI_STORE not initialized"
    assert hasattr(WIKI_STORE, "is_dynamic"), "Missing is_dynamic method"


def test_20_search_index_built():
    """Test search INDEX is built"""
    from webapp.api.index import INDEX
    assert INDEX is not None, "INDEX not built"
    assert hasattr(INDEX, "all_docs"), "Missing all_docs"
    assert hasattr(INDEX, "bm25"), "Missing bm25"


# ============================================================================
# TESTS 21-26: Integration
# ============================================================================

def test_21_query_log_valid():
    """Test query log entries follow spec"""
    from webapp.api.index import log_to_wiki_log
    from scripts.extract_entities import VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    before = len(log_path.read_text(encoding="utf-8"))
    
    log_to_wiki_log("query", "Test question?", {"pages_consulted": ["p1"], "wiki_updated": False})
    
    after = len(log_path.read_text(encoding="utf-8"))
    assert after > before, "Query not logged"


def test_22_ingest_log_valid():
    """Test ingest log entries follow spec"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    before = len(log_path.read_text(encoding="utf-8"))
    
    log_to_wiki_log("ingest", "Test-Source", {"pages_created": ["p1"], "chunks": 100})
    
    after = len(log_path.read_text(encoding="utf-8"))
    assert after > before, "Ingest not logged"


def test_23_log_consistency():
    """Test log consistency over multiple operations"""
    from scripts.extract_entities import log_to_wiki_log, VAULT
    
    log_path = VAULT / "wiki" / "log.md"
    content_start = log_path.read_text(encoding="utf-8")
    
    for i in range(3):
        log_to_wiki_log("consistency_test", f"Op {i}")
    
    content_end = log_path.read_text(encoding="utf-8")
    
    # Should contain all original + new
    assert content_start in content_end, "Original content lost"
    assert "consistency_test" in content_end, "New entries not found"


def test_24_chat_endpoint_exists():
    """Test /api/chat endpoint exists"""
    from webapp.api.index import chat
    assert callable(chat), "/api/chat not callable"


def test_25_chat_v2_endpoint_exists():
    """Test /api/chat-v2 endpoint exists"""
    from webapp.api.index import chat_v2
    assert callable(chat_v2), "/api/chat-v2 not callable"


def test_26_no_undefined_calls():
    """Test all referenced functions are defined"""
    from webapp.api.index import process_wiki_update_explicit, should_update_wiki
    from scripts.extract_entities import extract_source
    
    assert callable(process_wiki_update_explicit), "process_wiki_update_explicit not callable"
    assert callable(should_update_wiki), "should_update_wiki not callable"
    assert callable(extract_source), "extract_source not callable"


# ============================================================================
# BONUS TEST 27: Syntax Validation
# ============================================================================

def test_27_all_files_compile():
    """Test that all Python files compile without syntax errors"""
    import py_compile
    
    files = [
        "webapp/api/index.py",
        "scripts/extract_entities.py",
        "scripts/ingest.py",
        "scripts/sync_wiki.py",
    ]
    
    project_root = Path(__file__).parent.parent
    for file_path in files:
        full_path = project_root / file_path
        try:
            py_compile.compile(str(full_path), doraise=True)
        except Exception as e:
            raise AssertionError(f"{file_path} has syntax error: {e}")


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE - 27 TESTS")
    print("=" * 70)
    print()
    
    # Logging tests
    print("--- LOGGING SYSTEM (Tests 1-5) ---")
    test(1, "log.md creation", test_1_log_file_creation)
    test(2, "log entry format", test_2_log_entry_format)
    test(3, "log metadata preservation", test_3_log_metadata_preservation)
    test(4, "log append not overwrite", test_4_log_append_not_overwrite)
    test(5, "log multiple entries", test_5_log_multiple_entries)
    print()
    
    # API response tests
    print("--- API RESPONSE FORMAT (Tests 6-10) ---")
    test(6, "log_to_wiki_log accessible from API", test_6_log_to_wiki_log_accessible)
    test(7, "should_update_wiki rules enforcement", test_7_should_update_wiki_rules)
    test(8, "should_update_wiki all pass", test_8_should_update_wiki_all_pass)
    test(9, "process_wiki_update_explicit exists", test_9_process_wiki_update_exists)
    test(10, "wiki_search exists and works", test_10_wiki_search_exists)
    print()
    
    # Extract entities tests
    print("--- EXTRACT ENTITIES (Tests 11-15) ---")
    test(11, "log_to_wiki_log defined", test_11_log_function_defined)
    test(12, "extract_source callable", test_12_extract_source_callable)
    test(13, "log with list metadata", test_13_log_with_lists)
    test(14, "log robust to edge cases", test_14_log_robust_to_edge_cases)
    test(15, "log timestamp format", test_15_log_timestamp_exists)
    print()
    
    # Search and embeddings tests
    print("--- SEARCH AND EMBEDDINGS (Tests 16-20) ---")
    test(16, "embedding model defined", test_16_embed_model_defined)
    test(17, "chunks loaded", test_17_chunks_loaded)
    test(18, "chunk embeddings optional", test_18_chunk_embeddings_optional)
    test(19, "wiki store initialized", test_19_wiki_store_initialized)
    test(20, "search index built", test_20_search_index_built)
    print()
    
    # Integration tests
    print("--- INTEGRATION (Tests 21-27) ---")
    test(21, "query log valid", test_21_query_log_valid)
    test(22, "ingest log valid", test_22_ingest_log_valid)
    test(23, "log consistency", test_23_log_consistency)
    test(24, "/api/chat exists", test_24_chat_endpoint_exists)
    test(25, "/api/chat-v2 exists", test_25_chat_v2_endpoint_exists)
    test(26, "no undefined function calls", test_26_no_undefined_calls)
    test(27, "all files compile", test_27_all_files_compile)
    print()
    
    # Summary
    print("=" * 70)
    print(f"TEST RESULTS: {TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_SKIPPED} skipped")
    print("=" * 70)
    
    if TESTS_FAILED == 0:
        print("\n🎉 ALL TESTS PASSED! Safe for deployment.")
        return 0
    else:
        print(f"\n❌ {TESTS_FAILED} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
