"""Keep native simulator diagnostics from interrupting console progress."""

from contextlib import contextmanager
import os
import shutil
import sys
import tempfile


@contextmanager
def quiet_native_logs():
    """Capture native stdout/stderr for one operation; replay them if it raises."""
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as captured:
        saved = [os.dup(1), os.dup(2)]
        failed = False
        try:
            os.dup2(captured.fileno(), 1)
            os.dup2(captured.fileno(), 2)
            yield
        except BaseException:
            failed = True
            raise
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for target, original in zip((1, 2), saved):
                os.dup2(original, target)
                os.close(original)
            if failed:
                captured.seek(0)
                with os.fdopen(os.dup(2), "wb") as error_stream:
                    shutil.copyfileobj(captured, error_stream)
