import msprime
import random 
import matplotlib.pyplot as plt
import demesdraw
import tskit
from IPython.display import SVG, display
import os
import math
import summary_statistics as ss
import pandas as pd
import json
import numpy as np



def sim_replicates(simu_settings, num_replicates, modeldemo, T_1, T_2):

    n_1 = msprime.SampleSet(num_samples=simu_settings['n_1'], population="D1", time=0, ploidy=2)
    n_2 = msprime.SampleSet(num_samples=simu_settings['n_2'], population="D2", time=0, ploidy=2)
    
    n_1_T1 = msprime.SampleSet(num_samples=simu_settings['n_1_T1'], population="D1", time=T_1, ploidy=2)
    n_2_T1 = msprime.SampleSet(num_samples=simu_settings['n_2_T1'], population="D2", time=T_1, ploidy=2)
    
    n_1_T2 = msprime.SampleSet(num_samples=simu_settings['n_1_T2'], population="D1", time=T_2, ploidy=2)
    n_2_T2 = msprime.SampleSet(num_samples=simu_settings['n_2_T2'], population="D2", time=T_2, ploidy=2)

    replicates = msprime.sim_ancestry(
        samples=[n_1, n_2, n_1_T1, n_2_T1, n_1_T2, n_2_T2],
        sequence_length = simu_settings['L'], 
        recombination_rate = simu_settings['rho'],
        demography=modeldemo,
        num_replicates = num_replicates)
    
    for ts in replicates:
        mutated_ts = msprime.sim_mutations(ts, rate=simu_settings['mu'])
        yield mutated_ts 

def simu_scenario(outpath, scenario, num_replicates, demo_settings, simu_settings, sumstats_settings, launch_settings):
    
    outpath_scenario = outpath+'/scenario_'+str(scenario)
    
    os.mkdir(outpath_scenario)
    
    N_1 = random.randint(demo_settings['N_1'][0],demo_settings['N_1'][1])
    r_1 = round(random.uniform(demo_settings['r_1'][0],demo_settings['r_1'][1]),6)

    N_2 = random.randint(demo_settings['N_2'][0],demo_settings['N_2'][1])
    r_2 = round(random.uniform(demo_settings['r_2'][0],demo_settings['r_2'][1]),6)

    T_split = random.randint(demo_settings['T_split'][0],demo_settings['T_split'][1])
    
    if T_split>1:
        T_1 = random.randint(1,T_split-1)
        T_2 = random.randint(T_1,T_split-1) 
    else :
        T_1 = 1
        T_2 = 1           
    
    start_migration = 0
    if T_split > 0:
        end_migration = random.randint(0, T_split - 1)
    else:
        end_migration = 0
    
    m1 = round(random.uniform(demo_settings['migration_rate'][0],
                                          demo_settings['migration_rate'][1]),10)                               
    m2 = m1                                        
                                                                   
    N_ancestral = random.randint(demo_settings['N_ancestral'][0],demo_settings['N_ancestral'][1])

    modeldemo = msprime.Demography()

    modeldemo.add_population(
        name = "D1",
        initial_size = N_1,
        growth_rate = r_1)

    modeldemo.add_population(
        name = "D2",
        initial_size = N_2,
        growth_rate = r_2)

    modeldemo.add_population(
        name="Ancestral", 
        initial_size = N_ancestral)
   
    modeldemo.add_migration_rate_change(time=start_migration, rate=m1, source="D2", dest="D1")  
    modeldemo.add_migration_rate_change(time=start_migration, rate=m2, source="D1", dest="D2") 
    
    modeldemo.add_migration_rate_change(time=end_migration, rate=0, source="D1", dest="D2")  
    modeldemo.add_migration_rate_change(time=end_migration, rate=0, source="D2", dest="D1") 
  
    modeldemo.add_migration_rate_change(time=T_split, rate=0, source="D1", dest="D2") 
    modeldemo.add_migration_rate_change(time=T_split, rate=0, source="D2", dest="D1") 
    
    modeldemo.add_population_split(time=T_split, derived=["D1", "D2"], ancestral="Ancestral")  
          
    sim_replicates(simu_settings, num_replicates, modeldemo, T_1, T_2)
    
    S_l, PI_mean_l, PI_std_l, D_l, winHET_l, winHET_std_l, sfs_list_l, sfs_dist_list_l, LD_mean_list_l, LD_var_list_l, IBS_q_list_l, AFIBS_list_l, Dxy_l, Fst_l, Jsfs_list_l = ([] for i in range(15))
    list_ss = [S_l, PI_mean_l, PI_std_l, D_l, winHET_l, winHET_std_l, sfs_list_l, sfs_dist_list_l, LD_mean_list_l, LD_var_list_l, IBS_q_list_l, AFIBS_list_l, Dxy_l, Fst_l, Jsfs_list_l]
    list_name_ss = ["S", "PI_mean", "PI_std", "D", "winHET", "winHET_std", "sfs", "sfs_dist", "LD_mean", "LD_var", "IBS_q", "AFIBS", "Dxy", "Fst", "Jsfs"]
    list_rep = []

    for replicate_index, ts in enumerate(sim_replicates(simu_settings, num_replicates, modeldemo, T_1, T_2)):
        path = outpath_scenario +'/'+'ts_'+str(scenario)+'_'+str(replicate_index)
        if launch_settings['tree_saving']:
            ts.dump(path)
            
        if launch_settings['ss_computation']:
            ss_replicate = ss.sumstats(ts, simu_settings, sumstats_settings)
        
            for i in range(len(list_ss)):
                list_ss[i].append(ss_replicate[i])
            list_rep.append(replicate_index)
       
    if launch_settings['ss_saving'] and not launch_settings['ss_computation']:
        df_ss = pd.DataFrame()
        list_name_param = ['N_ancestral', 'N_1', 'r_1','m1', 'N_2', 'r_2','m2', 'T_split', 'end_migration', 'start_migration', 'n_1', 'n_2', 'L', 'rho', 'mu']
        list_val_param = [N_ancestral, N_1, r_1, m1, N_2, r_2, m2, T_split, end_migration, start_migration,  simu_settings['n_1'], simu_settings['n_2'], simu_settings['L'], simu_settings['rho'], simu_settings['mu']]
        df_ss['scenario_idx'] = [str(scenario)] * num_replicates
        if list_rep:
            df_ss['replicate_idx'] = list_rep
        for i in range(len(list_name_param)):
            df_ss[list_name_param[i]] = list_val_param[i]
        name_output = 'sumstats_scenario_' + str(scenario)
        df_ss.to_csv(outpath_scenario + '/' + name_output + ".csv", index=False)

    elif launch_settings['ss_computation']:        
        df_ss = pd.DataFrame()
        for i in range(len(list_ss)):
            df = pd.DataFrame(list_ss[i])
            col_names = [list_name_ss[i] + '_' + str(j) for j in range(df.shape[1])]
            df.columns = col_names
            df_ss = pd.concat([df_ss, df], axis=1)
        df_ss = pd.concat([pd.DataFrame(list_rep, columns=["replicate_idx"]), df_ss], axis=1)
        df_ss.insert(0, 'scenario_idx', str(scenario))
        list_name_param = ['N_ancestral', 'N_1', 'r_1','m1', 'N_2', 'r_2','m2', 'T_split', 'end_migration', 'start_migration', 'n_1', 'n_2', 'L', 'rho', 'mu']
        list_val_param = [N_ancestral, N_1, r_1, m1, N_2, r_2, m2, T_split, end_migration, start_migration,  simu_settings['n_1'], simu_settings['n_2'], simu_settings['L'], simu_settings['rho'], simu_settings['mu']]
        for i in range(len(list_name_param)):
            df_ss.insert(i+2, list_name_param[i], list_val_param[i])    
        if launch_settings['ss_saving']:
            name_output = 'sumstats_scenario_' + str(scenario)
            df_ss.to_csv(outpath_scenario + '/' + name_output + ".csv", index=False)

    return f"Scenario {scenario} completed"

