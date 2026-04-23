import numpy as np
from sklearn.neighbors import NearestNeighbors

# This class covered_states is used to record the covered states
# by the current testing input.

# Its current implementation is based on the tensorfFuzz
# https://github.com/brain-research/tensorfuzz/blob/master/lib/corpus.py

_BUFFER_SIZE = 5
_INIT_SIZE = 1

class CoveredStates(object):
    """Class holding the state of the update function."""

    def __init__(self, threshold=0.50, algorithm="kd_tree"):
        """Inits the object.
        Args:
          threshold: Float distance at which coverage is considered new.
          algorithm: NearestNeighbors algorithm (e.g., 'kd_tree', 'ball_tree', 'brute').
        Returns:
          Initialized object.
        """
        self.threshold = threshold
        self.algorithm = algorithm
        self.corpus_buffer = []
        self.lookup_array = []
        self.nn_index = None

        self.corpus = []

    def build_index_and_flush_buffer(self):
        """Builds the nearest neighbor index and flushes buffer of examples.
        This method first empties the buffer of examples that have not yet
        been added to the nearest neighbor index.
        Then it rebuilds that index using the contents of the whole corpus.
        Args:
          corpus_object: InputCorpus object.
        """
        self.corpus_buffer[:] = []
        self.lookup_array = np.vstack(
            self.corpus
        )
        self.nn_index = NearestNeighbors(n_neighbors=1, algorithm=self.algorithm, metric="euclidean")
        self.nn_index.fit(self.lookup_array)
        # tf.logging.info("Flushing buffer and building index.")

    def update_function(self, element):
        """Checks if coverage is new and updates corpus if so.
        The updater maintains both a corpus_buffer and a lookup_array.
        When the corpus_buffer reaches a certain size, we empty it out
        and rebuild the nearest neighbor index.
        Whenever we check for neighbors, we get exact neighbors from the
        buffer and approximate neighbors from the index.
        This stops us from building the index too frequently.
        We keep a small in-memory buffer and periodically rebuild the index,
        following the same high-level strategy as the original implementation.
        Args:
          corpus_object: InputCorpus object.
          element: CorpusElement object to maybe be added to the corpus.
        """
        if len(self.corpus) == 0:
            # print("waiting for element")
            self.corpus.append(element)
            self.build_index_and_flush_buffer()
            return True, 100
        else:
            if self.nn_index is None:
                self.build_index_and_flush_buffer()

            distances, _ = self.nn_index.kneighbors(np.asarray(element).reshape(1, -1), n_neighbors=1)
            approx_distances = [float(distances[0][0] ** 2)]
            exact_distances = [
                np.sum(np.square(element - buffer_elt))
                for buffer_elt in self.corpus_buffer
            ]
            nearest_distance = min(exact_distances + approx_distances)
            if nearest_distance > self.threshold:
                # tf.logging.info(
                #     "corpus_size %s mutations_processed %s",
                #     len(corpus_object.corpus),
                #     corpus_object.mutations_processed,
                # )
                # tf.logging.info(
                #     "coverage: %s, metadata: %s",
                #     element.coverage,
                #     element.metadata,
                # )
                self.corpus.append(element)
                self.corpus_buffer.append(element)
                if len(self.corpus_buffer) >= _BUFFER_SIZE:
                    self.build_index_and_flush_buffer()
                return True, nearest_distance
            else:
                return False, nearest_distance
