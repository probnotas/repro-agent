"""Claim JSON parsing: clean, fenced, prose-wrapped, malformed, and lossy."""

from __future__ import annotations

import json

import pytest

from repro.extract import Claim, ExtractionError, parse_claim, strip_fences


def test_parses_clean_json(claim_json: str) -> None:
    claim = parse_claim(claim_json)
    assert isinstance(claim, Claim)
    assert claim.environment == "Pendulum-v1"
    assert claim.reported_value == -150.0
    assert claim.specified["seeds"] is True
    assert claim.unspecified == []


def test_parses_json_fences(claim_json: str) -> None:
    assert parse_claim(f"```json\n{claim_json}\n```").reported_value == -150.0


def test_parses_bare_fences(claim_json: str) -> None:
    assert parse_claim(f"```\n{claim_json}\n```").environment == "Pendulum-v1"


def test_parses_json_wrapped_in_prose(claim_json: str) -> None:
    noisy = f"Sure! Here is the claim:\n\n{claim_json}\n\nLet me know if you need more."
    assert parse_claim(noisy).metric == "mean_episode_return"


def test_parses_fences_plus_prose(claim_json: str) -> None:
    noisy = f"Here you go:\n```json\n{claim_json}\n```\nHope that helps."
    assert parse_claim(noisy).reported_value == -150.0


def test_malformed_json_raises_with_preview() -> None:
    with pytest.raises(ExtractionError) as excinfo:
        parse_claim('{"claim_text": "x", "reported_value": }')
    message = str(excinfo.value)
    assert "not valid JSON" in message
    assert "claim_text" in message  # the preview shows what came back


def test_empty_response_raises() -> None:
    with pytest.raises(ExtractionError, match="empty response"):
        parse_claim("   \n  ")


def test_prose_only_response_raises() -> None:
    with pytest.raises(ExtractionError, match="not valid JSON"):
        parse_claim("I could not find a testable claim in this paper.")


def test_json_array_raises() -> None:
    with pytest.raises(ExtractionError, match="Expected a JSON object"):
        parse_claim("[1, 2, 3]")


def test_missing_claim_text_raises(claim_payload: dict) -> None:
    claim_payload.pop("claim_text")
    with pytest.raises(ExtractionError, match="no 'claim_text'"):
        parse_claim(json.dumps(claim_payload))


def test_blank_claim_text_raises(claim_payload: dict) -> None:
    claim_payload["claim_text"] = "   "
    with pytest.raises(ExtractionError, match="no 'claim_text'"):
        parse_claim(json.dumps(claim_payload))


def test_reported_value_as_string_is_coerced(claim_payload: dict) -> None:
    claim_payload["reported_value"] = "about -200.5 return"
    assert parse_claim(json.dumps(claim_payload)).reported_value == -200.5


def test_reported_value_unparseable_becomes_none(claim_payload: dict) -> None:
    claim_payload["reported_value"] = "substantially better"
    assert parse_claim(json.dumps(claim_payload)).reported_value is None


def test_reported_value_null_stays_none(claim_payload: dict) -> None:
    claim_payload["reported_value"] = None
    assert parse_claim(json.dumps(claim_payload)).reported_value is None


def test_specified_defaults_to_false_when_absent(claim_payload: dict) -> None:
    claim_payload.pop("specified")
    claim = parse_claim(json.dumps(claim_payload))
    assert claim.specified == {
        "hyperparameters": False,
        "seeds": False,
        "env_version": False,
        "success_threshold": False,
    }
    assert len(claim.unspecified) == 4


def test_specified_partial_fills_the_rest(claim_payload: dict) -> None:
    claim_payload["specified"] = {"seeds": True}
    claim = parse_claim(json.dumps(claim_payload))
    assert claim.specified["seeds"] is True
    assert claim.unspecified == ["hyperparameters", "env_version", "success_threshold"]


def test_missing_given_as_string_becomes_a_list(claim_payload: dict) -> None:
    claim_payload["missing"] = "the learning rate"
    assert parse_claim(json.dumps(claim_payload)).missing == ["the learning rate"]


def test_missing_given_as_null_becomes_empty(claim_payload: dict) -> None:
    claim_payload["missing"] = None
    assert parse_claim(json.dumps(claim_payload)).missing == []


def test_strip_fences_recovers_the_object() -> None:
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('prefix {"a": 1} suffix') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_shipped_fixtures_parse(testable_claim, untestable_claim) -> None:
    assert testable_claim.reported_value == -210.0
    assert untestable_claim.reported_value is None
    assert len(untestable_claim.unspecified) == 4
