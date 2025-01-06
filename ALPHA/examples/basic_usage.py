"""Basic example demonstrating ALPHA system usage."""

from ALPHA.core.interface import ALPHACore


def main():
    """Basic usage example of ALPHA system."""
    alpha = ALPHACore("example")
    
    print("\n1. Processing simple function...")
    result = alpha.process("""
def calculate_sum(numbers: list[int]) -> int:
    return sum(numbers)
""")
    print("Result:", result)
    
    print("\n2. Processing class with multiple methods...")
    result = alpha.process("""
class DataAnalyzer:
    def __init__(self, data: list[float]):
        self.data = data
    
    def mean(self) -> float:
        return sum(self.data) / len(self.data)
    
    def variance(self) -> float:
        mean = self.mean()
        squared_diff = [(x - mean) ** 2 for x in self.data]
        return sum(squared_diff) / len(self.data)
""")
    print("Result:", result)
    
    print("\n3. Checking system state...")
    state = alpha.get_state()
    print("System State:", state)
    
    print("\n4. Running optimization...")
    metrics = alpha.optimize()
    print("Optimization Metrics:", metrics)
    
    print("\n5. Processing optimized code...")
    result = alpha.process("""
def optimized_function(x: int, y: int) -> int:
    # This should be optimized based on previous patterns
    return x * y + x
""")
    print("Result:", result)


if __name__ == "__main__":
    main() 