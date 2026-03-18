from msgflux.utils.xml import apply_xml_tags


def test_apply_xml_tags():
    assert apply_xml_tags("tag", "content") == "<tag>content</tag>"


def test_apply_xml_tags_with_output_id():
    assert apply_xml_tags("tag", "content", "output_tag") == "<tag>content</output_tag>"
