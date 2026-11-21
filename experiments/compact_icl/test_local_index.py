"""Check streamed local embeddings against their expected cosine retrieval."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.vector_db.local_index import LocalValueIndex, write_local_index_column, write_local_index_manifest


class StreamedIndexTests(unittest.TestCase):
    def test_batches_preserve_normalization_and_cpu_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = write_local_index_column(root, "names", "name", ["first", "second", "zero"],
                                             iter([[[3., 4.]], [[0., 2.], [0., 0.]]]))
            array = np.load(root / entry["embeddings_file"], mmap_mode="c")
            self.assertIsInstance(array, np.memmap)
            np.testing.assert_allclose(array, [[.6, .8], [0., 1.], [0., 0.]], atol=1e-7)
            write_local_index_manifest(root, [entry])
            result = LocalValueIndex(root, device="cpu").retrieve_values_for_column(
                [[3., 4.]], "names", "name", 2, False
            )
            self.assertEqual([row["value"] for row in result["values"]], ["first", "second"])
            self.assertAlmostEqual(result["values"][0]["distance"], 0.)

    def test_incomplete_or_inconsistent_batches_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            for batches in ([[[1., 2.]]], [[[1., 2.]], [[1., 2., 3.]]]):
                with self.assertRaises(ValueError):
                    write_local_index_column(directory, "names", "name", ["a", "b"], iter(batches))


if __name__ == "__main__":
    unittest.main()
