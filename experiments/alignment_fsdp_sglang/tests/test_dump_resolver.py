from pathlib import Path

from utils.dump_resolver import FILE_RE


def test_dump_file_regex_matches_expected_name():
    name = "forward_pass_id=0___rank=0___name=layer0_q_post_norm___dump_index=12.pt"
    match = FILE_RE.match(name)
    assert match is not None
    assert match.group("name") == "layer0_q_post_norm"
    assert match.group("dump_index") == "12"


def test_path_type_is_usable():
    path = Path("/tmp/example")
    assert isinstance(path, Path)
