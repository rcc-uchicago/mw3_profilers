import cProfile
import pstats

def example_function():
    total = sum(i**2 for i in range(1000000))
    return total

cProfile.run('example_function()', 'output.prof')
stats = pstats.Stats('output.prof')
stats.strip_dirs().sort_stats('cumulative').print_stats(10)

