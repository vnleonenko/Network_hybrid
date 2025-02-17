N = 10**4
G = nx.barabasi_albert_graph(N, 5)#nx.complete_graph(N)

tmax = 250 
iterations = 30  # run 5 simulations 

tau = 0.01          # transmission rate 
alpha = 0.1 # latent period rate
gamma = 0.08   # recovery rate

rho = 0.005      # random fraction initially infected