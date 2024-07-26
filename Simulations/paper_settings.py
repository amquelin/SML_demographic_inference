demo_settings = {
    "N_ancestral" : [100, 10000],
    "N_1" : [100, 10000],
    "r_1" : [0,0],
    "N_2" : [100, 10000],
    "r_2" : [0,0],
    "migration_rate" : [0, 0.001],
    "T_split" : [1,5000],
} 

simu_settings = {
    "num_scenarios" : 10000,
    "num_replicates" : 20,
    "n_1" : 10,     #sample size for population A
    "n_2" : 10,     #sample size for population B 
    "L" : 2e6,      #sequence length
    "rho": 1e-8,  #recombination rate
    "mu": 1.25e-8   #mutation rate
}
