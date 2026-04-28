import warnings
import logging


class _MessageFilter(logging.Filter):
    def __init__(self, suppressed_text):
        super().__init__()
        self.suppressed_text = suppressed_text

    def filter(self, record):
        return self.suppressed_text not in record.getMessage()


def suppress_known_runtime_warnings():
    warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
    warnings.filterwarnings("ignore", message=".*Default qconfig of oneDNN backend.*")
    warnings.filterwarnings("ignore", message=".*torch\\.ao\\.quantization is deprecated.*")
    warnings.filterwarnings("ignore", message=".*triton not found; flop counting will not work.*")
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    logging.getLogger().addFilter(_MessageFilter("triton not found; flop counting will not work"))
