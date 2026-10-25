"""pytest -p no:xdist tests"""

import json

import numpy as np
import pytest

import labvespa

SD = """
# a comment with field fake type string {
schema fiqa {
  document fiqa {
    field id type string {
      indexing: attribute | summary
    }
    field dense_float type tensor<bfloat16>(x[4]) {
      indexing: attribute
      attribute {
        distance-metric: dotproduct
      }
    }
  }
  field body_pairs type string {
    indexing: input body | index
  }
  rank-profile hybrid {
    first-phase {
      expression: 1
    }
  }
}
"""


def test_fields_are_the_document_blocks_only():
    assert labvespa.fields(SD, "fiqa") == {"id": "string", "dense_float": "tensor<bfloat16>(x[4])"}
    with pytest.raises(labvespa.VespaError, match="no 'document caiq"):
        labvespa.fields(SD, "caiq")


def test_the_feed_fingerprint_moves_with_document_fields_only(tmp_path):
    (tmp_path / "schemas").mkdir()
    sd = tmp_path / "schemas" / "fiqa.sd"
    sd.write_text(SD)
    before = labvespa.fingerprint(tmp_path, "fiqa")
    sd.write_text(SD.replace("expression: 1", "expression: 2").replace("# a comment", "# another comment"))
    assert labvespa.fingerprint(tmp_path, "fiqa") == before
    sd.write_text(SD.replace("bfloat16", "float"))
    assert labvespa.fingerprint(tmp_path, "fiqa") != before


def test_hexed_writes_each_cell_type_big_endian():
    x = np.array([1.0, -2.0], dtype=np.float32)
    assert labvespa.hexed(x, "float") == "3f800000c0000000"
    assert labvespa.hexed(x, "bfloat16") == "3f80c000"
    assert labvespa.hexed(x, "double") == "3ff0000000000000c000000000000000"
    assert labvespa.hexed(np.array([-1, 127]), "int8") == "ff7f"


def test_the_two_schemas_differ_only_in_their_name():
    fiqa = (labvespa.APP / "schemas" / "fiqa.sd").read_text()
    caiq = (labvespa.APP / "schemas" / "caiq.sd").read_text()
    assert caiq == fiqa.replace("schema fiqa {", "schema caiq {").replace("document fiqa {", "document caiq {")


def test_the_feed_writes_the_declared_fields_and_warns_of_the_rest(monkeypatch):
    ds = labvespa.lab.Dataset("caiq", ["a", "b"], ["T\x10", ""], ["x", "y"], {}, {}, {})
    monkeypatch.setattr(labvespa.lab, "load", lambda name, split="test": ds)
    monkeypatch.setattr(labvespa.lab, "vectors", lambda ds: (np.array([[0.1, -0.2], [0.0, 0.25]], dtype=np.float32), None))
    warned = []
    n, lines = labvespa.documents("caiq", {"id": "string", "title": "string", "dense_int8": "tensor<int8>(x[2])",
                                           "mine": "string"}, warned.append)
    docs = [json.loads(line) for line in lines]
    assert n == 2 and warned == ["caiq.sd field mine: the lab has no values for it, so it stays empty"]
    assert docs[0] == {"put": "id:caiq:caiq::a", "fields": {"id": "a", "title": "T ", "dense_int8": {"values": "339a"}}}
    with pytest.raises(labvespa.VespaError, match="dense_int8 needs int8 cells"):
        labvespa.documents("caiq", {"dense_int8": "tensor<float>(x[2])"}, warned.append)
