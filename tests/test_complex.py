from hypothesis import given
from hypothesis.strategies import complex_numbers
from pytest import approx

from kalman_experiments.complex import complex2mat, complex2vec, mat2complex, vec2complex


@given(complex_numbers(max_magnitude=1e12))
def test_vec2complex_inverts_complex2vec(z: complex):
    assert vec2complex(complex2vec(z)) == z


@given(complex_numbers(max_magnitude=1e12))
def test_mat2complex_inverts_complex2mat(z: complex):
    assert mat2complex(complex2mat(z)) == z


@given(complex_numbers(max_magnitude=1e12), complex_numbers(max_magnitude=1e12))
def test_matrix_representation_delivers_correct_multiplication(z1: complex, z2: complex) -> None:
    M1 = complex2mat(z1)
    M2 = complex2mat(z2)
    assert mat2complex(M1 @ M2) == approx(z1 * z2, rel=1e-12)  # pyright: ignore
