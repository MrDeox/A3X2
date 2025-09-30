"""Simple calculator module for testing."""

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract two numbers."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    result = 0
    for i in range(b):
        result = add(result, a)
    return result

def divide(a, b):
    """Divide two numbers."""
    if b == 0:
        return "Error: Division by zero"
    result = 0
    while a >= b:
        a = subtract(a, b)
        result = add(result, 1)
    return result

def power(base, exp):
    """Calculate base to the power of exp."""
    result = 1
    counter = 0
    while counter < exp:
        result = multiply(result, base)
        counter = add(counter, 1)
    return result

def calculate_circle_area(radius):
    """Calculate the area of a circle."""
    pi = 3.14159  # Magic number
    return multiply(pi, multiply(radius, radius))

def calculate_rectangle_area(length, width):
    """Calculate the area of a rectangle."""
    global rect_counter  # Global variable
    rect_counter = rect_counter + 1 if 'rect_counter' in globals() else 1
    return multiply(length, width)

def calculate_triangle_area(base, height):
    """Calculate the area of a triangle."""
    return divide(multiply(base, height), 2)

# Test the functions
if __name__ == "__main__":
    print("Testing calculator functions:")
    print(f"Add: {add(5, 3)}")
    print(f"Subtract: {subtract(5, 3)}")
    print(f"Multiply: {multiply(5, 3)}")
    print(f"Divide: {divide(15, 3)}")
    print(f"Power: {power(2, 3)}")
    print(f"Circle area: {calculate_circle_area(5)}")
    print(f"Rectangle area: {calculate_rectangle_area(4, 6)}")
    print(f"Triangle area: {calculate_triangle_area(3, 4)}")