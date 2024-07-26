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


def sim_replicates(simu_settings, num_replicates, modeldemo):
    
    replicates = msprime.sim_ancestry(
        samples={"D1": simu_settings['n_1'], "D2": simu_settings['n_2']}, 
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
    
    # Demographic model

    N_1 = random.randint(demo_settings['N_1'][0],demo_settings['N_1'][1])
    r_1 = round(random.uniform(demo_settings['r_1'][0],demo_settings['r_1'][1]),6)

    N_2 = random.randint(demo_settings['N_2'][0],demo_settings['N_2'][1])
    r_2 = round(random.uniform(demo_settings['r_2'][0],demo_settings['r_2'][1]),6)

    T_split = random.randint(demo_settings['T_split'][0],demo_settings['T_split'][1])

    migration_rate = round(random.uniform(demo_settings['migration_rate'][0],
                                          demo_settings['migration_rate'][1]),10)

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

    modeldemo.add_population_split(
        time = T_split, 
        derived=["D1", "D2"], 
        ancestral="Ancestral")

    modeldemo.set_symmetric_migration_rate(["D1", "D2"], migration_rate)
        
    sim_replicates(simu_settings, num_replicates, modeldemo)
    
    #We initialize lists for all summary statistics that will be incremented for each replicate
    S_l ,PI_mean_l , PI_std_l , D_l , winHET_l , winHET_std_l , sfs_list_l , sfs_dist_list_l , LD_mean_list_l , LD_var_list_l , IBS_q_list_l , AFIBS_list_l , Dxy_l , Fst_l , Jsfs_list_l = ([] for i in range(15))
    list_ss = [S_l ,PI_mean_l , PI_std_l , D_l , winHET_l , winHET_std_l , sfs_list_l , sfs_dist_list_l , LD_mean_list_l , LD_var_list_l , IBS_q_list_l , AFIBS_list_l , Dxy_l , Fst_l , Jsfs_list_l]
    #name of the summary statistics in the output file
    list_name_ss = ["S" ,"PI_mean" , "PI_std" , "D" , "winHET" , "winHET_std" , "sfs" , "sfs_dist" , "LD_mean" , "LD_var" , "IBS_q" , "AFIBS" , "Dxy" , "Fst" , "Jsfs"]
    #List that keeps track of the replicate index for the output file
    list_rep =[]

    for replicate_index, ts in enumerate(sim_replicates(simu_settings, num_replicates, modeldemo)):
        #there will be one directory per scenario
        path = outpath_scenario +'/'+'ts_'+str(scenario)+'_'+str(replicate_index)
        #Export the tree sequence for the given replicate
        if launch_settings['tree_saving']:
            ts.dump(path)
        #Compute the summary statistics for the tree sequence
        ss_replicate = ss.sumstats(ts, simu_settings, sumstats_settings)
        
        #We append the list of sumstats with the sumstats of the replicate
        for i in range(len(list_ss)):
            list_ss[i].append(ss_replicate[i])
        #Keep the replicate index
        list_rep.append(replicate_index)
        
    #dataframe that will store the data to be exported
    df_ss = pd.DataFrame()
    #We iterate on list_ss by creating dataframe that we concatenate to df_ss
    for i in range(len(list_ss)):
        df = pd.DataFrame(list_ss[i])
        col_names =[list_name_ss[i] + '_'+ str(j) for j in range(df.shape[1])]
        df.columns = col_names
        df_ss = pd.concat([df_ss,df], axis = 1)
    #add list of replicate to the df to export
    df_ss = pd.concat([pd.DataFrame(list_rep, columns=["replicate_idx"]), df_ss], axis = 1)
    #Keep the scenario index
    df_ss.insert(0,'scenario_idx',str(scenario))
    #Add scenario parameters
    list_name_param = ['N_ancestral', 'N_1', 'r_1', 'N_2', 'r_2','migration_rate', 'T_split','n_1','n_2','L','rho','mu']
    list_val_param = [N_ancestral, N_1, r_1, N_2, r_2, migration_rate, T_split, simu_settings['n_1'], simu_settings['n_2'], simu_settings['L'], simu_settings['rho'], simu_settings['mu']]
    for i in range(len(list_name_param)):
        df_ss.insert(i+2, list_name_param[i], list_val_param[i])    
    
    #export the sumstats at the scenario directory level
    if launch_settings['ss_saving']:
        name_output = 'sumstats_scenario_' + str(scenario)
        df_ss.to_csv(outpath_scenario  +'/'+ name_output + ".csv", index = False)        

    return f"Scenario {scenario} completed"    
        
