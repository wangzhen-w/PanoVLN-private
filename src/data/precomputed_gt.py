"""Join final instructions with the original trajectory actions, without replay."""

import gzip
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Sequence

from tqdm import tqdm


@contextmanager
def open_json_text(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        yield handle


class StreamingJSONReader:
    """Small standard-library streaming reader for arrays and object entries."""

    def __init__(self, handle, chunk_size: int = 1024 * 1024):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _read_more(self) -> bool:
        if self.eof:
            return False
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if not chunk:
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def _ensure_available(self) -> None:
        while self.position >= len(self.buffer):
            if not self._read_more():
                raise ValueError("Unexpected end of JSON input")

    def skip_whitespace(self) -> None:
        while True:
            self._ensure_available()
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position].isspace()
            ):
                self.position += 1
            if self.position < len(self.buffer):
                return

    def peek(self) -> str:
        self.skip_whitespace()
        return self.buffer[self.position]

    def expect(self, token: str) -> None:
        self.skip_whitespace()
        if self.buffer[self.position] != token:
            raise ValueError(
                f"Expected {token!r}, found {self.buffer[self.position]!r}"
            )
        self.position += 1

    def read_value(self):
        self.skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError as error:
                if not self._read_more():
                    raise ValueError("Invalid or truncated JSON input") from error


def iter_array_field(path: str, field_name: str) -> Iterator[Dict]:
    """Yield objects from a named top-level JSON array."""

    with open_json_text(path) as handle:
        reader = StreamingJSONReader(handle)
        reader.expect("{")
        while reader.peek() != "}":
            key = reader.read_value()
            if not isinstance(key, str):
                raise ValueError(f"Expected object key in {path}, got {key!r}")
            reader.expect(":")
            if key != field_name:
                reader.read_value()
            else:
                reader.expect("[")
                if reader.peek() == "]":
                    reader.expect("]")
                    return
                while True:
                    value = reader.read_value()
                    if not isinstance(value, dict):
                        raise ValueError(
                            f"{field_name} entries must be objects in {path}"
                        )
                    yield value
                    separator = reader.peek()
                    if separator == ",":
                        reader.expect(",")
                        continue
                    if separator == "]":
                        reader.expect("]")
                        return
                    raise ValueError(
                        f"Expected ',' or ']' in {field_name} array in {path}"
                    )

            separator = reader.peek()
            if separator == ",":
                reader.expect(",")
                continue
            if separator == "}":
                break
            raise ValueError(f"Expected ',' or '}}' in top-level object in {path}")

        raise ValueError(f"Missing top-level {field_name!r} array in {path}")


def extract_instruction_text(episode: Dict) -> str:
    instruction = episode.get("instruction")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(
            f"Episode {episode.get('episode_id')} has no instruction text"
        )
    return instruction.strip()


def validate_actions(episode_id: int, raw_actions) -> list:
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError(f"Episode {episode_id} has no GT actions")

    actions = []
    for action in raw_actions:
        if isinstance(action, bool):
            raise ValueError(f"Episode {episode_id} contains boolean action")
        action_id = int(action)
        if action_id not in {0, 1, 2, 3}:
            raise ValueError(
                f"Episode {episode_id} contains unsupported action {action_id}"
            )
        actions.append(action_id)

    if actions[-1] != 0:
        raise ValueError(f"Episode {episode_id} GT actions do not end with stop")
    if 0 in actions[:-1]:
        raise ValueError(f"Episode {episode_id} contains an early stop action")
    return actions


def build_annotation(episode: Dict, trajectory: Dict) -> Dict:
    episode_id = int(episode["episode_id"])
    trajectory_id = episode.get("trajectory_id")
    if trajectory_id is None or isinstance(trajectory_id, bool):
        raise ValueError(f"Episode {episode_id} has no usable trajectory_id")
    if trajectory_id != trajectory.get("trajectory_id"):
        raise ValueError(f"Episode {episode_id} does not match its source trajectory")

    # Images use the numeric episode_id; the source trajectory hash is not an image ID.
    return {
        "episode_id": episode_id,
        "instruction": extract_instruction_text(episode),
        "actions": validate_actions(episode_id, trajectory.get("action_ids")),
    }


def write_precomputed_annotations(
    episode_path: str,
    trajectory_path: str,
    output_path: str,
    episode_ids: Optional[Sequence[int]] = None,
    max_episodes: Optional[int] = None,
) -> Dict[str, int]:
    """Match accepted episode IDs to indices in the complete trajectory file."""

    if max_episodes is not None and max_episodes < 0:
        raise ValueError("max_episodes must be non-negative")

    selected_ids = (
        None if episode_ids is None else {int(value) for value in episode_ids}
    )
    found_selected_ids = set()
    seen_episode_ids = set()
    seen_trajectory_ids = set()
    total_count = 0
    written_count = 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=output_dir or ".",
        text=True,
    )

    try:
        episode_iter = iter_array_field(episode_path, "episodes")
        trajectory_iter = enumerate(iter_array_field(trajectory_path, "episodes"))
        trajectory_index = -1
        previous_episode_id = -1
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_handle:
            for episode in tqdm(
                episode_iter,
                desc="import trajectory actions",
                unit="episode",
                dynamic_ncols=True,
            ):
                episode_id = int(episode["episode_id"])
                if episode_id <= previous_episode_id:
                    raise ValueError("Episode IDs must be non-negative and strictly increasing")
                previous_episode_id = episode_id
                while trajectory_index < episode_id:
                    try:
                        trajectory_index, trajectory = next(trajectory_iter)
                    except StopIteration as error:
                        raise ValueError(f"Missing source trajectory for episode {episode_id}") from error

                annotation = build_annotation(episode, trajectory)
                episode_id = annotation["episode_id"]
                trajectory_key = str(episode["trajectory_id"])
                if episode_id in seen_episode_ids:
                    raise ValueError(f"Duplicate episode_id: {episode_id}")
                if trajectory_key in seen_trajectory_ids:
                    raise ValueError(f"Duplicate trajectory_id: {trajectory_key}")
                seen_episode_ids.add(episode_id)
                seen_trajectory_ids.add(trajectory_key)
                total_count += 1

                if selected_ids is not None:
                    if episode_id not in selected_ids:
                        continue
                    found_selected_ids.add(episode_id)
                if max_episodes is not None and written_count >= max_episodes:
                    continue

                output_handle.write(
                    json.dumps(annotation, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                written_count += 1

            missing_ids = (
                set() if selected_ids is None else selected_ids - found_selected_ids
            )
            if missing_ids:
                raise ValueError(
                    f"Missing requested episode ids: {sorted(missing_ids)}"
                )

            output_handle.flush()
            os.fsync(output_handle.fileno())

        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return {
        "source_episodes": total_count,
        "written_episodes": written_count,
    }
