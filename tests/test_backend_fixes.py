import tempfile
import unittest

from persistent_memory import PersistentMemory
from prompt_engineering import needs_grounding_repair


class CacheKeyStabilityTests(unittest.TestCase):
    """Regression test: the cache key must be based on durable memory, not
    the constantly-changing recent conversation — otherwise a verbatim
    repeated question never hits cache once any turn has happened."""

    def test_cache_key_is_stable_across_conversation_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            user_id = "user-1"

            key_before = memory.cache_key(
                user_id, "What is RAG?", "doc-v1", "gemini:model", memory.memories_text(user_id)
            )

            # An intervening conversation turn changes recent_turns_text...
            memory.add_turn(user_id, "Tell me about transformers.", "They use self-attention.")

            key_after = memory.cache_key(
                user_id, "What is RAG?", "doc-v1", "gemini:model", memory.memories_text(user_id)
            )

            self.assertEqual(key_before, key_after)

    def test_cache_key_changes_when_durable_memory_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            user_id = "user-1"

            key_before = memory.cache_key(
                user_id, "What is RAG?", "doc-v1", "gemini:model", memory.memories_text(user_id)
            )
            memory.add_memory(user_id, "I prefer concise answers.")
            key_after = memory.cache_key(
                user_id, "What is RAG?", "doc-v1", "gemini:model", memory.memories_text(user_id)
            )

            self.assertNotEqual(key_before, key_after)


class GroundingRepairTests(unittest.TestCase):
    def test_short_answer_with_evidence_needs_repair(self):
        self.assertTrue(needs_grounding_repair("Yes.", "Source: paper.pdf, page 2\nSome real evidence."))

    def test_answer_without_citation_needs_repair(self):
        self.assertTrue(
            needs_grounding_repair(
                "The model uses self-attention across all positions in the sequence.",
                "Source: paper.pdf, page 2\nSome real evidence.",
            )
        )

    def test_answer_with_page_citation_does_not_need_repair(self):
        self.assertFalse(
            needs_grounding_repair(
                "According to page 2, the model uses self-attention.",
                "Source: paper.pdf, page 2\nSome real evidence.",
            )
        )

    def test_no_retrieved_context_never_needs_repair(self):
        self.assertFalse(needs_grounding_repair("Short.", "No relevant passages were found in the PDF."))

    def test_inline_page_citation_format_is_recognized(self):
        # Regression test: prompts now ask for inline citations formatted
        # like "(paper.pdf, p. 4)" rather than the word "page" — the
        # grounding check must recognize that format too, not just "page N".
        self.assertFalse(
            needs_grounding_repair(
                "Self-attention is used across positions (paper.pdf, p. 2).",
                "Source: paper.pdf, page 2\nSome real evidence.",
            )
        )


class PromptEngineeringCitationFormatTests(unittest.TestCase):
    """Regression test: every prompt that asks a model to cite evidence
    must ask for the same inline citation format, so answers from every
    provider are consistent and `needs_grounding_repair` can recognize them."""

    def test_all_prompts_request_the_inline_citation_format(self):
        from prompt_engineering import agent_system_prompt, ollama_prompt, repair_prompt

        for text in (
            agent_system_prompt(),
            ollama_prompt("<task>context</task>"),
            repair_prompt("<task>context</task>", "draft answer"),
        ):
            self.assertIn("p. N", text)


if __name__ == "__main__":
    unittest.main()
