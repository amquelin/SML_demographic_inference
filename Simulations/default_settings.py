
#SETTINGS #

import numpy as np

launch_settings = {
    #tree sequence exports
    "tree_saving" : False,
    #sumstats computation
    "ss_computation" : True,    
    #csv exports
    "ss_saving" : True
}

demo_settings = {
    "N_ancestral" : [1000, 5000],
    "N_1" : [2000, 2000],
    "r_1" : [0,0],
    "N_2" : [2000, 2000],
    "r_2" : [0,0],
    "migration_rate" : [0, 0.001],
    "T_split" : [10,2000],
} 

simu_settings = {
    "num_scenarios" : 100,
    "num_replicates" : 20,
    "n_1" : 10,     #sample size for population A
    "n_2" : 10,     #sample size for population B 
    "L" : 2e6,      #sequence length
    "rho": 1e-8,  #recombination rate
    "mu": 1.25e-8   #mutation rate
}

sumstats_settings = {
    "win_size_hh" :  50e3,
    # LD settings:
    "nb_times": 21,
    "Tmax": 130000,
    "a": 0.06,
    "per_err": 5,   # percentage for defining lower and upper bounds relative to a distance
    # IBS settings:
    "size_list" : [2, 4, 8, 16],
    "prob_list" : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}

# LD settings
# The older Tmax, the shorter the minimal distance at which LD is calculated
# From Boitard et al. 2016: nb_times=21; Tmax=130000; a=0.06; per_err=5
nb_times = sumstats_settings["nb_times"]
Tmax = sumstats_settings["Tmax"]
a = sumstats_settings["a"]
per_err = sumstats_settings["per_err"]  # percentage for defining lower and upper bounds relative to a distance
times = -np.ones(shape=nb_times, dtype="float")
for i in range(nb_times):
    times[i] = (np.exp(np.log(1+a*Tmax)*i/(nb_times-1))-1)/a
# Define bins (of distance) for which to LD stats will be computed
interval_list = []
for i in range(nb_times-1):
    t = (times[i+1]+times[i])/2
    d = 10**8/(2*t)
    if d <= simu_settings["L"]:
        interval_list.append([d-per_err*d/100, d+per_err*d/100])
t = Tmax + times[nb_times-1] - times[nb_times-2]
d = 10**8 / (2*t)
interval_list.append([d-per_err*d/100, d+per_err*d/100])

sumstats_settings["times"] = times
sumstats_settings["interval_list"] = interval_list  # intervals for distance bins for LD
