from functools import partial, reduce

def iterate(f, n):
    iterator = iter(f)  # Convert the iterable into an iterator
    for _ in range(n):         # Call `next` n times
        result = next(iterator)
    return result

jump = partial(iterate, f=next)