from typing import Any, Dict, List, Optional

import torch


STACKABLE_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "image_erp_geometry",
    "image_num_images",
    "image_current_index",
    "panovggt_pixel_values",
)

class MultiModalDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
        }

        has_attention_mask = all("attention_mask" in feature for feature in features)
        if has_attention_mask:
            attention_mask = [feature["attention_mask"] for feature in features]
            batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            )
        else:
            batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id)

        if all("mm_token_type_ids" in feature for feature in features):
            mm_token_type_ids = [feature["mm_token_type_ids"] for feature in features]
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm_token_type_ids,
                batch_first=True,
                padding_value=0,
            )

        for key in STACKABLE_KEYS:
            present_count = sum(key in feature for feature in features)
            if present_count == 0:
                continue
            if present_count != len(features):
                raise ValueError(
                    f"Batch has inconsistent multimodal field '{key}': "
                    f"{present_count}/{len(features)} samples include it"
                )
            batch[key] = torch.cat([feature[key] for feature in features], dim=0)

        return batch
