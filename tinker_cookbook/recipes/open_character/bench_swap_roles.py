"""
Microbenchmark for swap_roles implementations.

Compares 5 implementations:
1. Original - Explicit loop building new list
2. Dict one-liner - List comprehension with dict spread
3. In-place - Mutate and swap back manually
4. Context manager (simple) - Inline swap logic
5. Context manager (clear) - Nested swap_inplace() for clarity

Usage:
    python -m tinker_cookbook.recipes.open_character.bench_swap_roles
"""

import timeit
import tracemalloc
from contextlib import contextmanager

ROLE_SWAP = {"user": "assistant", "assistant": "user"}


# =============================================================================
# Implementation 1: Original (explicit loop, new list)
# =============================================================================

def swap_roles_original(messages: list[dict]) -> list[dict]:
    swapped = []
    for msg in messages:
        role = msg["role"]
        if role == "user":
            swapped.append({"role": "assistant", "content": msg["content"]})
        elif role == "assistant":
            swapped.append({"role": "user", "content": msg["content"]})
        else:
            swapped.append(msg.copy())
    return swapped


# =============================================================================
# Implementation 2: Dict one-liner (list comprehension)
# =============================================================================

def swap_roles_oneliner(messages: list[dict]) -> list[dict]:
    return [{**m, "role": ROLE_SWAP.get(m["role"], m["role"])} for m in messages]


# =============================================================================
# Implementation 3: In-place (manual swap back)
# =============================================================================

def swap_roles_inplace(messages: list[dict]) -> None:
    for msg in messages:
        msg["role"] = ROLE_SWAP.get(msg["role"], msg["role"])


# =============================================================================
# Implementation 4: Context manager (simple - inline logic)
# =============================================================================

@contextmanager
def swapped_roles_simple(messages: list[dict]):
    for msg in messages:
        msg["role"] = ROLE_SWAP.get(msg["role"], msg["role"])
    try:
        yield messages
    finally:
        for msg in messages:
            msg["role"] = ROLE_SWAP.get(msg["role"], msg["role"])


# =============================================================================
# Implementation 5: Context manager (clear - nested function)
# =============================================================================

@contextmanager
def swapped_roles_clear(messages: list[dict]):
    def swap_inplace(msgs: list[dict]) -> None:
        for msg in msgs:
            msg["role"] = ROLE_SWAP.get(msg["role"], msg["role"])
    
    swap_inplace(messages)
    try:
        yield messages
    finally:
        swap_inplace(messages)


# =============================================================================
# Benchmark helpers
# =============================================================================

def make_messages(n: int) -> list[dict]:
    """Generate n alternating user/assistant messages with realistic content."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(n - 1):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"This is message {i} with some typical content that might appear in a conversation."
        messages.append({"role": role, "content": content})
    return messages


def use_result(messages: list[dict]) -> None:
    """Prevent dead code elimination by 'using' the result."""
    _ = len(messages)


def bench_cpu_original(messages: list[dict], iterations: int) -> float:
    """Benchmark original implementation."""
    def fn():
        result = swap_roles_original(messages)
        use_result(result)
    
    total = timeit.timeit(fn, number=iterations)
    return (total / iterations) * 1_000_000  # microseconds


def bench_cpu_oneliner(messages: list[dict], iterations: int) -> float:
    """Benchmark one-liner implementation."""
    def fn():
        result = swap_roles_oneliner(messages)
        use_result(result)
    
    total = timeit.timeit(fn, number=iterations)
    return (total / iterations) * 1_000_000


def bench_cpu_inplace(messages: list[dict], iterations: int) -> float:
    """Benchmark in-place implementation (with manual swap back)."""
    def fn():
        swap_roles_inplace(messages)
        use_result(messages)
        swap_roles_inplace(messages)  # swap back
    
    total = timeit.timeit(fn, number=iterations)
    return (total / iterations) * 1_000_000


def bench_cpu_ctx_simple(messages: list[dict], iterations: int) -> float:
    """Benchmark simple context manager."""
    def fn():
        with swapped_roles_simple(messages) as swapped:
            use_result(swapped)
    
    total = timeit.timeit(fn, number=iterations)
    return (total / iterations) * 1_000_000


def bench_cpu_ctx_clear(messages: list[dict], iterations: int) -> float:
    """Benchmark clear context manager."""
    def fn():
        with swapped_roles_clear(messages) as swapped:
            use_result(swapped)
    
    total = timeit.timeit(fn, number=iterations)
    return (total / iterations) * 1_000_000


def bench_memory_original(messages: list[dict]) -> int:
    """Measure memory for original implementation."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    result = swap_roles_original(messages)
    use_result(result)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bench_memory_oneliner(messages: list[dict]) -> int:
    """Measure memory for one-liner implementation."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    result = swap_roles_oneliner(messages)
    use_result(result)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bench_memory_inplace(messages: list[dict]) -> int:
    """Measure memory for in-place implementation."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    swap_roles_inplace(messages)
    use_result(messages)
    swap_roles_inplace(messages)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bench_memory_ctx_simple(messages: list[dict]) -> int:
    """Measure memory for simple context manager."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    with swapped_roles_simple(messages) as swapped:
        use_result(swapped)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bench_memory_ctx_clear(messages: list[dict]) -> int:
    """Measure memory for clear context manager."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    with swapped_roles_clear(messages) as swapped:
        use_result(swapped)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main():
    print("Swap Roles Microbenchmark")
    print("=" * 60)
    
    sizes = [10, 50, 200, 2000]
    
    for n in sizes:
        messages = make_messages(n)
        
        # Adjust iterations based on size for reasonable runtime
        iterations = 10000 if n <= 200 else 1000
        
        print(f"\nMessages: {n}")
        print(f"  {'Implementation':<20} {'CPU (μs)':>12} {'Memory (bytes)':>16}")
        
        # Original
        cpu = bench_cpu_original(messages, iterations)
        mem = bench_memory_original(messages)
        print(f"  {'original':<20} {cpu:>12.2f} {mem:>16,}")
        
        # One-liner
        cpu = bench_cpu_oneliner(messages, iterations)
        mem = bench_memory_oneliner(messages)
        print(f"  {'dict_oneliner':<20} {cpu:>12.2f} {mem:>16,}")
        
        # In-place
        cpu = bench_cpu_inplace(messages, iterations)
        mem = bench_memory_inplace(messages)
        print(f"  {'inplace':<20} {cpu:>12.2f} {mem:>16,}")
        
        # Context manager (simple)
        cpu = bench_cpu_ctx_simple(messages, iterations)
        mem = bench_memory_ctx_simple(messages)
        print(f"  {'ctx_simple':<20} {cpu:>12.2f} {mem:>16,}")
        
        # Context manager (clear)
        cpu = bench_cpu_ctx_clear(messages, iterations)
        mem = bench_memory_ctx_clear(messages)
        print(f"  {'ctx_clear':<20} {cpu:>12.2f} {mem:>16,}")
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("  - CPU: Average time per call in microseconds (lower is better)")
    print("  - Memory: Peak allocation in bytes (lower is better)")
    print("  - In-place variants should show ~0 memory allocation")


if __name__ == "__main__":
    main()
