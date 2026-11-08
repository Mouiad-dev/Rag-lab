import pytest
from pydantic import ValidationError
from rag_foundry_techniques import TextBlock


def test_optional_structure_fields_preserve_old_construction():
    block = TextBlock(block_id="b00000", start=0, end=1)
    assert block.kind is None and block.section_path == () and block.heading_level is None and block.page_index is None


@pytest.mark.parametrize("kwargs", [{"heading_level": 2}, {"kind": "heading"}, {"kind": "heading", "heading_level": 7}])
def test_heading_level_consistency(kwargs):
    with pytest.raises(ValidationError):
        TextBlock(block_id="b00000", start=0, end=1, **kwargs)
