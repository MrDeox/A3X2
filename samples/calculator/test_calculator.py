"""Tests for the calculator module."""

from .calculator import (
    add,
    calculate_circle_area,
    calculate_rectangle_area,
    calculate_triangle_area,
    divide,
    multiply,
    power,
    subtract,
)


def test_add():
    """Test addition function."""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0


def test_subtract():
    """Test subtraction function."""
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5
    assert subtract(10, 10) == 0


def test_multiply():
    """Test multiplication function."""
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6


def test_divide():
    """Test division function."""
    assert divide(10, 2) == 5
    assert divide(15, 3) == 5
    assert "Error" in divide(10, 0)  # Test division by zero


def test_power():
    """Test power function."""
    assert power(2, 3) == 8
    assert power(5, 0) == 1
    assert power(3, 2) == 9


def test_calculate_circle_area():
    """Test circle area calculation."""
    # Approximate value due to pi approximation
    assert abs(calculate_circle_area(1) - 3.14159) < 0.0001


def test_calculate_rectangle_area():
    """Test rectangle area calculation."""
    assert calculate_rectangle_area(4, 5) == 20
    assert calculate_rectangle_area(0, 5) == 0


def test_calculate_triangle_area():
    """Test triangle area calculation."""
    assert calculate_triangle_area(4, 6) == 12
    assert calculate_triangle_area(0, 5) == 0
