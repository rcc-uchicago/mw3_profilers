from memory_profiler import profile
#import pstats

@profile

def example_function():
    total = sum([i**2 for i in range(1000000)])
    return total
output = example_function()
#cProfile.run('example_function()', 'output.prof')
#stats = pstats.Stats('output.prof')
#stats.strip_dirs().sort_stats('cumulative').print_stats(10)

