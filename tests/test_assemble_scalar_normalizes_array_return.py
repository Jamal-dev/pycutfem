import numpy as np


class _DummyFunctional:
    def __init__(self):
        self.integrand = object()


def test_assemble_scalar_accepts_length_one_arrays(monkeypatch) -> None:
    from examples.utils.biofilm import adhesion

    monkeypatch.setattr(adhesion, "Equation", lambda *args, **kwargs: object())

    def fake_assemble_form(*args, **kwargs):
        return {"val": np.asarray([3.14])}

    monkeypatch.setattr(adhesion, "assemble_form", fake_assemble_form)
    val = adhesion.assemble_scalar(dof_handler=None, functional=_DummyFunctional(), backend="cpp", quad_order=1)
    assert float(val) == 3.14


def test_assemble_scalar_accepts_scalar_like(monkeypatch) -> None:
    from examples.utils.biofilm import adhesion

    monkeypatch.setattr(adhesion, "Equation", lambda *args, **kwargs: object())

    def fake_assemble_form(*args, **kwargs):
        return {"val": np.asarray(2.5)}

    monkeypatch.setattr(adhesion, "assemble_form", fake_assemble_form)
    val = adhesion.assemble_scalar(dof_handler=None, functional=_DummyFunctional(), backend="cpp", quad_order=1)
    assert float(val) == 2.5
