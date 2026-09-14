"""Bounded episode concurrency with Habitat confined to its owning thread."""

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from itertools import groupby
from queue import Empty, Queue
from threading import Event


class RenderProxy:
    """An episode submits render work; it never accesses the simulator itself."""

    def __init__(self, episode, requests, stopped):
        self.episode = episode
        self.requests = requests
        self.stopped = stopped

    def _call(self, method, *args):
        future = Future()
        self.requests.put((self.episode, method, args, future))
        while not self.stopped.is_set():
            try:
                return future.result(timeout=.2)
            except TimeoutError:
                if future.done():
                    raise
        raise CancelledError("Rendering worker stopped")

    def replay(self, episode):
        return self._call("replay", episode)

    def ground_route(self, states):
        return self._call("ground_route", states)

    def render_segment(self, *args):
        return self._call("render_segment", *args)

    def export_training(self, *args):
        return self._call("export_training", *args)


def run_scene_tasks(items, renderer, concurrency, process, report):
    """Overlap API work on a bounded set of episodes, reusing one scene at a time.

    Call this on the thread that owns ``renderer``. Episode threads keep their
    own HTTP clients and records; all Habitat operations are served here.
    """
    requests, stopped = Queue(), Event()
    current_episode = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        try:
            for _, scene_items in groupby(items, key=lambda item: item[1]["scene_id"]):
                remaining = iter(scene_items)
                pending = set()

                def submit_next():
                    item = next(remaining, None)
                    if item is not None:
                        proxy = RenderProxy(item[1], requests, stopped)
                        pending.add(pool.submit(process, item, proxy))

                for _ in range(concurrency):
                    submit_next()
                while pending:
                    try:
                        episode, method, args, future = requests.get(timeout=.05)
                    except Empty:
                        pass
                    else:
                        try:
                            if current_episode != episode["trajectory_id"]:
                                renderer.memory_frames.clear()
                                current_episode = episode["trajectory_id"]
                            renderer.load_scene(episode["scene_id"])
                            value = getattr(renderer, method)(*args)
                        except Exception as error:
                            future.set_exception(error)
                        else:
                            future.set_result(value)
                    for future in [f for f in pending if f.done()]:
                        pending.remove(future)
                        report(future.result())
                        submit_next()
        finally:
            stopped.set()
