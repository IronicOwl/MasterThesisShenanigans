import matplotlib
matplotlib.use("Agg")  # safe for parallel/non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count

print("Hello Eric")

def make_plot(freq):
    x = np.linspace(1, 10, 100)
    y = np.sin(freq * x)
    plt.figure()
    plt.plot(x, y)
    plt.title(f"sin({freq}x)")
    plt.savefig(f"figure_freq_{freq}.png")
    plt.close()
    return freq

if __name__ == "__main__":
    freqs = [1, 2, 3, 4, 5]  # the "loop" values, now run in parallel

    with Pool(processes=min(len(freqs), cpu_count())) as pool:
        results = pool.map(make_plot, freqs)

    print("Done with frequencies:", results)
    
