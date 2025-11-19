# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Create the function.py file
COPY <<EOF /app/function.py
def calc(a1: int, a2: int):
    """Must return sum of two integers"""
    return a1 + a2
EOF

# Create the test file
COPY <<EOF /app/test_function.py
import unittest
from function import calc

class TestCalcFunction(unittest.TestCase):
    def test_positive_integers(self):
        """Test addition of positive integers"""
        self.assertEqual(calc(2, 3), 5)
        self.assertEqual(calc(10, 20), 30)
        self.assertEqual(calc(1, 1), 2)
    
    def test_negative_integers(self):
        """Test addition of negative integers"""
        self.assertEqual(calc(-2, -3), -5)
        self.assertEqual(calc(-10, -20), -30)
        self.assertEqual(calc(-1, -1), -2)
    
    def test_mixed_integers(self):
        """Test addition of positive and negative integers"""
        self.assertEqual(calc(5, -3), 2)
        self.assertEqual(calc(-10, 20), 10)
        self.assertEqual(calc(-1, 1), 0)
    
    def test_zero_values(self):
        """Test addition with zero"""
        self.assertEqual(calc(0, 0), 0)
        self.assertEqual(calc(0, 5), 5)
        self.assertEqual(calc(5, 0), 5)
        self.assertEqual(calc(0, -5), -5)
        self.assertEqual(calc(-5, 0), -5)
    
    def test_large_integers(self):
        """Test addition of large integers"""
        self.assertEqual(calc(1000000, 2000000), 3000000)
        self.assertEqual(calc(-1000000, -2000000), -3000000)
    
    def test_type_handling(self):
        """Test that the function works with integers"""
        # These should work fine
        self.assertEqual(calc(5, 3), 8)
        
        # Test with floats (should still work but might not be intended)
        self.assertEqual(calc(5.0, 3.0), 8.0)
        self.assertEqual(calc(5.5, 3.5), 9.0)

if __name__ == '__main__':
    unittest.main()
EOF

# Command to run the tests
CMD ["python", "-m", "unittest", "test_function.py"]