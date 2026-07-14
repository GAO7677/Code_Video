from code_vjepa_vggt.utils.physical_caption_phrases import extract_physical_noun_phrases

import numpy as np
from types import SimpleNamespace

from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
    physical_phrase_grounding_queries,
)


def _texts(caption: str) -> list[str]:
    selected, _ = extract_physical_noun_phrases(caption, max_phrases=4)
    return [item.text.lower() for item in selected]


def test_ball_block_caption() -> None:
    caption = (
        "Two pillows on a table and two grabber tools hanging above them from which "
        "a brown tennis ball and an orange block are suspended. The grabber tools "
        "let go of the ball and block. Static shot with no camera movement."
    )
    texts = _texts(caption)
    assert "brown tennis ball" in texts, texts
    assert "orange block" in texts, texts
    assert all("camera" not in item and "shot" not in item for item in texts)


def test_rotating_cardstock_caption() -> None:
    caption = (
        "A grabber arm is holding a tennis ball above a piece of cardstock propped up "
        "on a rotating platform sitting on a table that rotates clockwise. The grabber "
        "lowers the ball and places is on the table as the cardstock rotates. Static "
        "shot with no camera movement."
    )
    texts = _texts(caption)
    assert "tennis ball" in texts, texts
    assert "piece of cardstock" in texts, texts
    assert "rotating platform" in texts, texts
    assert "table" in texts, texts


def test_explicit_instance_count() -> None:
    selected, _ = extract_physical_noun_phrases(
        "Two red balls fall onto a platform while an orange block remains still.",
        max_phrases=4,
    )
    records = {item.text.lower(): item for item in selected}
    assert records["red balls"].instance_count == 2, records
    assert records["orange block"].instance_count == 1, records


def test_distinct_same_class_phrases_are_not_merged() -> None:
    selected, _ = extract_physical_noun_phrases(
        "A red ball and a blue ball collide while the ball remains visible.",
        max_phrases=4,
    )
    texts = {item.text.lower() for item in selected}
    assert "red ball" in texts, texts
    assert "blue ball" in texts, texts
    assert "ball" not in texts, texts


def test_cardstock_grounding_alias_preserves_source_phrase() -> None:
    selected, _ = extract_physical_noun_phrases(
        "A ball falls behind a piece of cardstock as the cardstock rotates.",
        max_phrases=4,
    )
    record = next(item for item in selected if item.head == "cardstock")
    assert physical_phrase_grounding_queries(record) == ["piece of cardstock", "paper"]


def test_provider_preserves_source_phrase_and_span() -> None:
    caption = (
        "A moving brown tennis ball collides with an orange block while a rotating "
        "platform supports both objects."
    )
    boxes = {
        "moving brown tennis ball": np.asarray([[10, 10, 30, 30]], dtype=np.float32),
        "orange block": np.asarray(
            [[40, 10, 65, 35], [5, 50, 20, 65], [70, 50, 90, 70]],
            dtype=np.float32,
        ),
        "rotating platform": np.asarray([[15, 45, 70, 62]], dtype=np.float32),
    }

    class FakeDetector:
        def detect(self, frame, phrase, guidance_box_xyxy=None):
            phrase_boxes = boxes.get(phrase, np.zeros((0, 4), dtype=np.float32))
            return SimpleNamespace(
                boxes_xyxy=phrase_boxes,
                scores=np.full((len(phrase_boxes),), 0.9, dtype=np.float32),
                phrases=["detector_label"] * len(phrase_boxes),
            )

    class FakeTracker:
        def track(self, frames, prompt_frame_idx, prompt_box_xyxy, caption=""):
            masks = np.zeros((len(frames), 80, 96), dtype=np.uint8)
            x0, y0, x1, y1 = prompt_box_xyxy.astype(int)
            masks[:, y0:y1, x0:x1] = 1
            return SimpleNamespace(
                masks_thw=masks,
                boxes_t4=np.repeat(prompt_box_xyxy[None], len(frames), axis=0),
            )

    provider = ViewerGroundingBoxProvider(
        device="cpu",
        segment_len=4,
        max_objects=4,
        points_per_object=2,
        proposal_source="gdino_only",
        motion_score_ratio=0.15,
        text_prompt="",
        extra_prompt_terms="",
        include_caption_terms=True,
        gdino_box_threshold=0.2,
        gdino_text_threshold=0.15,
        prompt_frame_mode="first",
        track_dedupe_iou_threshold=0.75,
        container_suppress_ratio_threshold=0.95,
        container_suppress_min_contained=2,
        container_suppress_min_area_ratio=1.5,
        container_suppress_small_iou_threshold=0.7,
        caption_prompt_mode="physical_noun_phrases",
        caption_max_phrases=4,
        caption_min_score=4.0,
    )
    provider.detector = FakeDetector()
    provider.tracker = FakeTracker()
    sample = provider.build_sample(
        frames_tchw_01=np.zeros((4, 3, 80, 96), dtype=np.float32),
        caption=caption,
        image_hw=(80, 96),
    )
    phrases = [track.phrase for track in sample.object_tracks]
    assert set(phrases) == {
        "moving brown tennis ball", "orange block", "rotating platform"
    }, phrases
    assert all(track.caption_span is not None for track in sample.object_tracks)
    assert sample.debug["caption_phrase_extraction"]["selected"]


if __name__ == "__main__":
    test_ball_block_caption()
    test_rotating_cardstock_caption()
    test_explicit_instance_count()
    test_distinct_same_class_phrases_are_not_merged()
    test_cardstock_grounding_alias_preserves_source_phrase()
    test_provider_preserves_source_phrase_and_span()
    print("physical caption phrase tests passed")
