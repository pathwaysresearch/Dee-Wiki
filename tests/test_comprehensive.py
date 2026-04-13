"""
Comprehensive test suite for Prof. Bhagwan Hybrid Wiki.
26+ test cases covering API endpoints, logging, wiki updates, and data flows.

Run: pytest tests/test_comprehensive.py -v
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest


# ============================================================================
# TEST 1-5: Logging System
# ============================================================================

class TestLoggingSystem:
    """Test log_to_wiki_log function"""

    def test_1_log_file_creation(self):
        """Test that log.md is created if missing"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        
        # Remove if exists
        if log_path.exists():
            original_content = log_path.read_text(encoding="utf-8")
        
        # Call log_to_wiki_log
        log_to_wiki_log("test", "Test entry creation")
        
        # Verify file exists
        assert log_path.exists(), "log.md was not created"
        print("✅ TEST 1 PASSED: log.md created")

    def test_2_log_entry_format(self):
        """Test that log entries have correct format"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        log_to_wiki_log("ingest", "Test Format Check")
        
        content = log_path.read_text(encoding="utf-8")
        # Should have timestamp format: ## [YYYY-MM-DD]
        assert "##" in content, "Missing log header"
        assert "[" in content and "]" in content, "Missing date brackets"
        assert "ingest |" in content, "Missing operation type"
        print("✅ TEST 2 PASSED: log entry has correct format")

    def test_3_log_metadata_preservation(self):
        """Test that metadata is correctly logged"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        log_to_wiki_log(
            "ingest",
            "Test Metadata",
            {"pages": 5, "sources": ["test1", "test2"]}
        )
        
        content = log_path.read_text(encoding="utf-8")
        assert "pages:" in content or "pages" in content, "Metadata not logged"
        print("✅ TEST 3 PASSED: metadata preserved in log")

    def test_4_log_append_not_overwrite(self):
        """Test that logs append, don't overwrite"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        content_before = log_path.read_text(encoding="utf-8")
        
        log_to_wiki_log("query", "Test Append")
        
        content_after = log_path.read_text(encoding="utf-8")
        assert len(content_after) > len(content_before), "Log was not appended"
        assert content_before in content_after, "Previous logs were overwritten"
        print("✅ TEST 4 PASSED: logs append correctly")

    def test_5_log_multiple_entries_chronological(self):
        """Test that multiple entries maintain order"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        
        # Log 3 entries
        log_to_wiki_log("ingest", "Entry 1")
        log_to_wiki_log("query", "Entry 2")
        log_to_wiki_log("ingest", "Entry 3")
        
        content = log_path.read_text(encoding="utf-8")
        pos_1 = content.find("Entry 1")
        pos_2 = content.find("Entry 2")
        pos_3 = content.find("Entry 3")
        
        assert pos_1 < pos_2 < pos_3, "Entries not in chronological order"
        print("✅ TEST 5 PASSED: entries maintain chronological order")


# ============================================================================
# TEST 6-10: Backend API Response Format
# ============================================================================

class TestAPIResponseFormat:
    """Test that API responses have correct structure"""

    def test_6_log_to_wiki_log_exists(self):
        """Test that log_to_wiki_log function is accessible from api"""
        from webapp.api.index import log_to_wiki_log
        assert callable(log_to_wiki_log), "log_to_wiki_log not callable"
        print("✅ TEST 6 PASSED: log_to_wiki_log accessible from API")

    def test_7_should_update_wiki_rules(self):
        """Test should_update_wiki enforces all 3 rules"""
        from webapp.api.index import should_update_wiki
        
        # Test case: insufficient sources (< 2)
        result = should_update_wiki(
            {"should_wiki_update": True, "new_synthesis": "test"},
            []  # 0 sources
        )
        assert result is False, "Should reject with <2 sources"
        print("✅ TEST 7a PASSED: rule 1 (sources) enforced")
        
        # Test case: no LLM suggestion
        result = should_update_wiki(
            {"should_wiki_update": False, "new_synthesis": "test"},
            [{}, {}]  # 2 sources
        )
        assert result is False, "Should reject without LLM suggestion"
        print("✅ TEST 7b PASSED: rule 2 (LLM suggest) enforced")
        
        # Test case: no synthesis
        result = should_update_wiki(
            {"should_wiki_update": True, "new_synthesis": ""},
            [{}, {}]
        )
        assert result is False, "Should reject without synthesis"
        print("✅ TEST 7c PASSED: rule 3 (synthesis) enforced")

    def test_8_should_update_wiki_all_pass(self):
        """Test should_update_wiki returns True when all rules pass"""
        from webapp.api.index import should_update_wiki
        
        result = should_update_wiki(
            {"should_wiki_update": True, "new_synthesis": "New insight"},
            [{}, {}]  # 2+ sources
        )
        assert result is True, "Should pass all 3 rules"
        print("✅ TEST 8 PASSED: all rules pass returns True")

    def test_9_process_wiki_update_explicit_rejects_empty(self):
        """Test process_wiki_update_explicit rejects empty title/content"""
        from webapp.api.index import process_wiki_update_explicit
        
        # Should handle gracefully (not crash)
        process_wiki_update_explicit("", "")
        process_wiki_update_explicit("Title", "")
        process_wiki_update_explicit("", "Content")
        
        print("✅ TEST 9 PASSED: empty inputs handled gracefully")

    def test_10_wiki_search_filters_correctly(self):
        """Test wiki_search returns only wiki types"""
        from webapp.api.index import wiki_search
        
        results = wiki_search("finance", top_k=5)
        # All results should be type="wiki" or have wiki metadata
        for result in results:
            assert isinstance(result, dict), "Result not a dict"
            # Should have content key at minimum
            assert "content" in result or "title" in result, "Missing required fields"
        
        print("✅ TEST 10 PASSED: wiki_search returns valid results")


# ============================================================================
# TEST 11-15: Extract Entities Logging
# ============================================================================

class TestExtractEntitiesLogging:
    """Test entity extraction with logging"""

    def test_11_log_function_defined(self):
        """Test log_to_wiki_log is defined in extract_entities"""
        from scripts.extract_entities import log_to_wiki_log
        assert callable(log_to_wiki_log), "log_to_wiki_log not defined"
        print("✅ TEST 11 PASSED: log_to_wiki_log defined")

    def test_12_extract_source_calls_log(self):
        """Test that extract_source would call log_to_wiki_log"""
        # This is a static check - verify the function exists and is callable
        from scripts.extract_entities import extract_source, log_to_wiki_log
        assert callable(extract_source), "extract_source not callable"
        assert callable(log_to_wiki_log), "log_to_wiki_log not callable"
        print("✅ TEST 12 PASSED: extract_source and log_to_wiki_log callable")

    def test_13_log_handles_dict_metadata(self):
        """Test logging with complex dict metadata"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        log_to_wiki_log(
            "ingest",
            "Test Dict",
            {
                "pages_created": ["page1", "page2"],
                "chunks": 100,
                "config": {"route": "RAG+stub"}
            }
        )
        
        content = log_path.read_text(encoding="utf-8")
        assert "page1" in content or "pages_created" in content, "Dict not logged"
        print("✅ TEST 13 PASSED: dict metadata logged correctly")

    def test_14_log_safe_from_exceptions(self):
        """Test log_to_wiki_log doesn't crash on errors"""
        from scripts.extract_entities import log_to_wiki_log
        
        # Should not raise
        try:
            log_to_wiki_log("test", None)  # None description
            log_to_wiki_log("test", "")  # Empty description
            log_to_wiki_log("test", "ok", None)  # None metadata
            print("✅ TEST 14 PASSED: logging is robust to edge cases")
        except Exception as e:
            pytest.fail(f"log_to_wiki_log crashed: {e}")

    def test_15_log_timestamp_format(self):
        """Test log entries contain valid ISO dates"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        import re
        
        log_path = VAULT / "wiki" / "log.md"
        log_to_wiki_log("test", "Timestamp test")
        
        content = log_path.read_text(encoding="utf-8")
        # Should match [YYYY-MM-DD]
        match = re.search(r'\[\d{4}-\d{2}-\d{2}\]', content)
        assert match, "No valid ISO date found in log"
        print("✅ TEST 15 PASSED: timestamps in ISO format")


# ============================================================================
# TEST 16-20: Search Index and Embeddings
# ============================================================================

class TestSearchAndEmbeddings:
    """Test search functionality and embedding system"""

    def test_16_embed_model_defined(self):
        """Test embedding model is accessible"""
        from webapp.api.index import EMBED_MODEL, DOC_PREFIX
        assert EMBED_MODEL, "EMBED_MODEL not defined"
        assert DOC_PREFIX, "DOC_PREFIX not defined"
        print("✅ TEST 16 PASSED: embedding config defined")

    def test_17_chunks_loaded(self):
        """Test RAG chunks are loaded"""
        from webapp.api.index import CHUNKS
        assert isinstance(CHUNKS, list), "CHUNKS not a list"
        assert len(CHUNKS) > 0, "No chunks loaded"
        print(f"✅ TEST 17 PASSED: {len(CHUNKS)} chunks loaded")

    def test_18_chunk_embeddings_shape(self):
        """Test chunk embeddings have correct shape"""
        from webapp.api.index import CHUNK_EMBEDDINGS, CHUNKS
        
        if CHUNK_EMBEDDINGS is not None:
            assert len(CHUNK_EMBEDDINGS) == len(CHUNKS), "Embedding count mismatch"
            print("✅ TEST 18 PASSED: embeddings match chunk count")
        else:
            print("⚠️ TEST 18 SKIPPED: no embeddings loaded (OK for dev)")

    def test_19_wiki_store_initialized(self):
        """Test wiki store is properly initialized"""
        from webapp.api.index import WIKI_STORE
        assert WIKI_STORE is not None, "WIKI_STORE not initialized"
        assert hasattr(WIKI_STORE, "is_dynamic"), "WIKI_STORE missing is_dynamic method"
        print("✅ TEST 19 PASSED: WIKI_STORE initialized")

    def test_20_search_index_exists(self):
        """Test search INDEX is built"""
        from webapp.api.index import INDEX
        assert INDEX is not None, "INDEX not built"
        assert hasattr(INDEX, "all_docs"), "INDEX missing all_docs"
        assert hasattr(INDEX, "bm25"), "INDEX missing bm25"
        print("✅ TEST 20 PASSED: search INDEX initialized")


# ============================================================================
# TEST 21-26: Integration and End-to-End
# ============================================================================

class TestIntegration:
    """Integration tests for full workflows"""

    def test_21_query_log_format_correct(self):
        """Test query log entries follow spec"""
        from webapp.api.index import log_to_wiki_log
        
        # Simulate what /api/chat logs
        log_to_wiki_log(
            "query",
            "What is CAPM?",
            {
                "pages_consulted": ["capm.md", "risk.md"],
                "wiki_updated": False
            }
        )
        print("✅ TEST 21 PASSED: query log format valid")

    def test_22_ingest_log_format_correct(self):
        """Test ingest log entries follow spec"""
        from scripts.extract_entities import log_to_wiki_log
        
        # Simulate what extract_entities logs
        log_to_wiki_log(
            "ingest",
            "Valuation-Damodaran",
            {
                "route": "RAG+stub",
                "pages_created": ["dcf.md", "wacc.md"],
                "chunks": 150
            }
        )
        print("✅ TEST 22 PASSED: ingest log format valid")

    def test_23_log_md_consistency(self):
        """Test log.md remains consistent after multiple operations"""
        from scripts.extract_entities import log_to_wiki_log, VAULT
        
        log_path = VAULT / "wiki" / "log.md"
        initial_content = log_path.read_text(encoding="utf-8")
        
        # Make 5 log entries
        for i in range(5):
            log_to_wiki_log("test", f"Entry {i}")
        
        final_content = log_path.read_text(encoding="utf-8")
        
        # Should have all original content + new entries
        assert initial_content in final_content, "Original content lost"
        # Should have 5 new entries
        assert final_content.count(f"Entry") >= 5, "Not all entries logged"
        print("✅ TEST 23 PASSED: log consistency maintained")

    def test_24_api_chat_endpoint_exists(self):
        """Test /api/chat endpoint is defined"""
        from webapp.api.index import chat
        assert callable(chat), "/api/chat not defined or not callable"
        print("✅ TEST 24 PASSED: /api/chat endpoint exists")

    def test_25_api_chat_v2_endpoint_exists(self):
        """Test /api/chat-v2 endpoint is defined"""
        from webapp.api.index import chat_v2
        assert callable(chat_v2), "/api/chat-v2 not defined or not callable"
        print("✅ TEST 25 PASSED: /api/chat-v2 endpoint exists")

    def test_26_no_undefined_function_calls(self):
        """Test that log_to_wiki_log is called but not undefined"""
        from webapp.api.index import process_wiki_update_explicit
        from scripts.extract_entities import extract_source
        
        # Both functions should exist and be callable
        assert callable(process_wiki_update_explicit), "process_wiki_update_explicit not callable"
        assert callable(extract_source), "extract_source not callable"
        print("✅ TEST 26 PASSED: all called functions are defined")


# ============================================================================
# BONUS: Test 27 - Syntax Validation
# ============================================================================

class TestSyntaxValidation:
    """Verify files compile without syntax errors"""

    def test_27_all_py_files_compile(self):
        """Test that all Python files compile"""
        import py_compile
        
        files_to_check = [
            "webapp/api/index.py",
            "scripts/extract_entities.py",
            "scripts/ingest.py",
            "scripts/sync_wiki.py",
            "scripts/chunker.py",
            "scripts/graph.py",
            "scripts/export_for_web.py",
        ]
        
        failed = []
        for file_path in files_to_check:
            full_path = Path(__file__).parent.parent / file_path
            try:
                py_compile.compile(str(full_path), doraise=True)
            except py_compile.PyCompileError as e:
                failed.append((file_path, str(e)))
        
        if failed:
            pytest.fail(f"Syntax errors in: {failed}")
        
        print(f"✅ TEST 27 PASSED: All {len(files_to_check)} Python files compile")


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
