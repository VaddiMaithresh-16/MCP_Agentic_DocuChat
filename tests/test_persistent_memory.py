import tempfile
import time
import unittest

from persistent_memory import PersistentMemory, ResponseCache


class PersistentMemoryTests(unittest.TestCase):
    def test_memory_turns_and_cache_survive_store_use(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            memory.add_memory("user-1", "I prefer short explanations.")
            memory.add_turn("user-1", "What is RAG?", "RAG combines retrieval and generation.")
            cache = ResponseCache(memory)
            cache.put("question-key", {"answer": "cached answer"})

            self.assertIn("short explanations", memory.context("user-1"))
            self.assertEqual(memory.recent_turns("user-1")[0]["question"], "What is RAG?")
            self.assertEqual(cache.get("question-key")["answer"], "cached answer")

    def test_personal_fact_is_captured_without_a_memory_name(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            captured = memory.auto_capture("user-1", "I prefer concise answers.")

            self.assertIsNotNone(captured)
            self.assertIn("concise answers", memory.memories("user-1")[0])

    def test_cache_entries_expire_after_ttl(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            cache = ResponseCache(memory, ttl_seconds=0)
            cache.put("key", {"answer": "stale soon"})
            time.sleep(0.01)
            self.assertIsNone(cache.get("key"))

    def test_cache_is_pruned_to_max_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = PersistentMemory(f"{directory}/memory.sqlite3")
            cache = ResponseCache(memory, max_entries=3)
            for index in range(6):
                cache.put(f"key-{index}", {"answer": f"answer-{index}"})
                time.sleep(0.001)  # ensure distinct created_at ordering

            with memory._connect() as connection:
                remaining = connection.execute("SELECT COUNT(*) AS n FROM response_cache").fetchone()["n"]
            self.assertEqual(remaining, 3)
            # the most recent entries should be the ones that survived
            self.assertIsNotNone(cache.get("key-5"))
            self.assertIsNone(cache.get("key-0"))


if __name__ == "__main__":
    unittest.main()
