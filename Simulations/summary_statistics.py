import numpy as np
import bisect
from collections import Counter
import scipy as sp
from scipy import stats
import default_settings

def het_one_win(dataslice,Nhap):
    '''
    Compute haplotypic heterozygosity of a given window

    Arguments:
    dataslice      np.array       subset of the data corresponding to a given window of the sequence
    Nhap           int            total number of haplotypes

    Return:
    het            float          haplotypic heterozygosity of dataslice
    '''

    haplos=[''.join([repr(num) for num in dataslice[i,:]]) for i in range(Nhap)]
    tab=Counter(haplos)
    return 1.0-sum([x**2 for x in list(tab.values())])/float(Nhap)**2 

def haplo_win(hap,pos,win_size,L=2000000):
    '''
    Compute haplotypic heterozygosity in windows sliding aloong the genome and return mean and variance

    Return:
    mean, std       float,float                      mean and standard deviation of haplotypic heterozygosity
    '''

    Nhap=hap.shape[0]
    L=int(L)
    win_size=int(win_size)
    hetsall=[]
    
    chunks=[bisect.bisect(pos,x) for x in range(0,L,win_size)]
    hets=[het_one_win(hap[:,chunks[i]:chunks[i+1]], Nhap) for i in range(len(chunks)-1)]
    hetsall.extend(hets)
             
    Nhap=float(Nhap)
    
    return np.array((Nhap/(Nhap-1.0) * np.mean(hets),  Nhap/(Nhap-1.0) * np.std(hets)))


def spatial_histo_fast(pos,count,M,dmax=np.inf):
    '''Computes the site frequency spectrum

    Fast version of spatial_histo
    Note: This is the correct implementation of dist
    From FJ

    Returns :
    - the site frequency spectrum from 1 to M (percentages, sum to 1)
    - the variance of the distance between sites with count i, for all i.
    '''
    histo=np.zeros(shape=M,dtype='float')
    nb_snp=0
    d = [[] for i in range(1,M+1) ]

    pos_c= [[] for i in range(1,M+1) ]
    for snp in range(len(pos)):
        i = count[snp].astype(int)
        try:
            histo[i-1]+=1
            pos_c[i-1].append(pos[snp])
        except IndexError:
            continue

    [d[i-1].append(x) for i in range (1,M+1) for x in np.diff(pos_c[i-1]) if x<=dmax]

    # for each frequency, compute the std of the distance list, after removing distance longer than dmax
        # if no distances deviation set to 0 (was -1 before)
    dist = np.asarray([np.std(d_at_freq) if len(d_at_freq)>1 else 0.0 for d_at_freq in d])
#    dist=np.asarray([np.std(d_at_freq) if len(d_at_freq)>1 else -1.0 for d_at_freq in d])

    # correct but with np.nan and 0 for len(d)=0 or 1
    # dist=[np.std([x for x in d_at_freq if x<=dmax]) for d_at_freq in d]
    
    return histo/np.sum(histo),dist

def r2(u,v):
    '''
    returns the r2 value for two haplotype vectors (numpy arrays with alleles coded 1 and 0)
    '''
    fcross=np.mean(u*v)
    fu=np.mean(u)
    fv=np.mean(v)
    return (fcross-fu*fv)**2/(fu*(1-fu)*fv*(1-fv))


def distrib_r2(pos,hap,interval_list):
    '''
    returns the mean and the variance of r2 for a list of distance intervals.
    pos_list is a list of 1 dim arrays
    hap_list is a list of 2 dim arrays
    interval_list is a list of ordered pairs
    a subset of non overlapping pairs is used for each interval
    From FJ
    '''
    p=len(interval_list)
    moy=-np.ones(shape=p,dtype='float')
    var=-np.ones(shape=p,dtype='float')
    
    for i in range(0,p):
        r2_list=[]
        dmin=interval_list[i][0]
        dmax=interval_list[i][1]
        # looks for snp pairs with the good distance

        nb_snp=len(pos)
        if nb_snp>0:
            i_deb=0
            i_fin=1
            while i_fin<nb_snp:
                while i_fin < nb_snp and (pos[i_fin]-pos[i_deb])<dmin:
                    i_fin+=1
                if i_fin < nb_snp and (pos[i_fin]-pos[i_deb])<=dmax and np.sum(hap[:,i_deb])<len(hap) and np.sum(hap[:,i_fin])<len(hap):
                    # compute r2
                    u_deb=hap[:,i_deb]
                    u_fin=hap[:,i_fin]
                    r2_list.append(r2(u_deb,u_fin))
                i_deb=i_fin+1
                i_fin=i_deb+1
                    
        if len(r2_list) < 2:
            # try a more exhaustive screening of SNP pairs
            r2_list=[]
            dmin=interval_list[i][0]
            dmax=interval_list[i][1]

            nb_snp=len(pos)
            if nb_snp>0:
                i_deb=0
                i_fin=1
                while i_fin<nb_snp:
                    while i_fin < nb_snp and (pos[i_fin]-pos[i_deb])<dmin:
                        i_fin+=1
                    if i_fin < nb_snp and (pos[i_fin]-pos[i_deb])<=dmax and np.sum(hap[:,i_deb])<len(hap) and np.sum(hap[:,i_fin])<len(hap):
                        # compute r2
                        u_deb=hap[:,i_deb]
                        u_fin=hap[:,i_fin]
                        r2_list.append(r2(u_deb,u_fin))
                    i_deb+=1
                    i_fin=i_deb+1
        # computes the stat
        if len(r2_list) >= 2:
            moy[i]=np.mean(np.array(r2_list,dtype='float'))
            var[i]=np.std(np.array(r2_list,dtype='float'))
    return moy,var


def ibs_quantiles_from_data(m,pos,data_type,data,prob_list,dmax=200000000,quantiles=False,moments=False):
    ''' Computes the quantiles of the ibs length distribution for a subset of m haplotypes or m diploid individuals

    WARNING: Set data_type to 1 if haplotypes, to 2 if genotypes
    if m==1 and data_type==2 this corresponds to ROH

    Arguments:
    m          int   nb of haplotypes or genotypes to randomly subsample
    pos_list   list(1 dim np.array)  positions of snps
    data_type  int                   if 1 data are haplotypes if 2 genotypes
    data_list  list(2-dim np.array)  haplotypes or genotypes
    prob_list  list(float)           vector of probabilities for which quantiles are computed
    dmax       int                   maximum length of ibs (eg. length of the segments), highly recommended to specify dmax
    quantiles  bool                  compute quantiles of ibs-length distribution?
    moments    bool                  compute moments of ibs-length distribution?

    Returns:
    np.concatenate((q,moms)) quantiles and moments 1 to 4th of ibs-length distrib

    From FJ from SB
    '''

    q=np.array([])
    moms=np.array([])
    # builds a ibs length sample
    d=np.zeros(shape=1,dtype='int32')
    
    pos_temp=pos
    data_temp=data
    n=data.shape[0]
    if m<n:
        # for each chromosome we randomly draw m haplotypes (or m genotypes)
        # so that we are not always using the same individuals for the computation
        # (it matters for real data)
        # and update count_temp and pos_temp
        subset=np.random.choice(n,size=m,replace=False)
        count_temp=np.sum(data_temp[subset,],axis=0)
        pos_temp=pos_temp[(count_temp>0)*(count_temp<(data_type*m))]
    if len(pos_temp)>1:
        d=np.concatenate((d,np.diff(pos_temp)))
    else:
        d=np.concatenate((d,dmax*np.ones(shape=1,dtype='int32')))
            
            
    d=np.minimum(d,dmax*np.ones(shape=len(d),dtype='int32'))
    # computes the quantiles and/or the moments of this sample
    if quantiles:
        q=sp.stats.mstats.mquantiles(d[1:],prob=prob_list,alphap=1,betap=1)
    if moments:
        moms=-np.ones(shape=4)
        moms[0]=np.mean(d[1:])
        moms[1]=np.std(d[1:])
        #mom3:skewness, mom4:kurtosis
        for m in range(3,5):
            moms[m-1]=np.mean(np.power((d[1:]-moms[0])/moms[1],m)) 
    return np.concatenate((q,moms))

def afibs_durbin_left_length(haplos,posOnChrom,counts,rightbounds,afibs):
    '''
    Compute afibs-leftbound and length of afibs segment using positions stored in posOnChrom
    Based on Durbin (PBWT), Bioinformatics 2014, Algorithm 2

    haplos        np.array[Nhap,Nsnp]    haplotype data for one region for Nhap individuals x Nsnp SNPs
    posOnChrom    int array      positions (bp) of each polymorphism relative to its chromosome
    counts        np.array[Nsnp_seg]         number of derived alleles at each position
    rightbounds   list of int [Nsnp]     SNP index of afibs-rightbound at each position returned by afibs_durbin_right
    afibs         list of list [Nhap]     list of afibs tract length to be updated 

    Returns:
    afibs         list of list [Nhap]     list containing a list for each allele count 0...Nhap

    Author: Flora Jay
    '''

    Nhap,Nsnp=haplos.shape
    # To keep the details of all segment lengths:
    if afibs is None:
        afibs = [[] for der in range(Nhap+1)]  

    # LEFT BOUND

    # Most variable names follow Durbin algo 2
    # But haplos replaces yi
    # d: divergence array
    # a: positional prefix array

    acurr= list(range(Nhap))
    dcurr= [0]*Nhap
    leftbounds=[0]*Nsnp

    for k in range(Nsnp):

        p,q=k+1,k+1
        a,b,d,e = [[] for i in range(4)]
        commonbound=0  # will contain the MAXIMAL index su ch that haplotypes carrying '1' at position k are all identical from commonbound (included) to k
        firstDerived=True
        for i in range(Nhap):
            if dcurr[i]>p:
                p=dcurr[i]
            if dcurr[i] >q:
                q=dcurr[i]
            if haplos[acurr[i],k]==0:
                a.append(acurr[i])
                d.append(p)
                p=0
            else:
                b.append(acurr[i])
                e.append(q)
                if firstDerived:
                    # When we encounter the first haplotype carrying a derived allele at k
                    # There is no bound to look for yet
                    # because the haplo differs from previous haplo at position k (previous haplo carries a '0')
                    firstDerived=False
                elif q>commonbound:
                    commonbound=q
                    # contains the current MAXIMAL index such that haplotypes already parsed and carrying '1' at position k are all identical from commonbound to k
                q=0
        acurr=a+b
        dcurr=d+e
        leftbounds[k] = commonbound-1    #try: np.max(e[1:])-1; except: pass


        # If left!=1 and right!=Nsnp then both bounds are inside the genomic region
        # and we can save the length of the segment
        # Otherwise it means they were not updated because the segment overlaps the region boundaries
        # so the exact length is not known and not saved
        if  (leftbounds[k]!=-1 and rightbounds[k]!=Nsnp):
            seglen=posOnChrom[rightbounds[k]]-posOnChrom[leftbounds[k]]
            afibs[counts[k]].append(seglen)
            # otherstat[counts[k]] += seglen**2  # if you want to directly compute other stats (eg for the moments, do it here and remember do the init and to add them to the return list)

    return afibs


def afibs_durbin_right(haplos):
    '''
    Search for afibs-rightbound at each SNP
    Based on Durbin (PBWT), Bioinformatics 2014


    Arg:
    haplos        np.array[Nhap,Nsnp]    haplotype data for one region for Nhap individuals x Nsnp SNPs

    Returns:
    rightbounds   list of int [Nsnp]     SNP index of afibs-rightbound at each position

    Author: Flora Jay
    '''

    Nhap,Nsnp=haplos.shape

    # Looking for RIGHT BOUND by applying Durbin algo starting the right
    # ie Loop on snp index starts at Nsnp and finishes at 0

    # Most variable names follow Durbin algo 2
    # But haplos replaces yi
    # d: divergence array
    # a: positional prefix array

    #start5=time.time()
    acurr= list(range(Nhap))
    dcurr= [Nsnp-1]*Nhap
    rightbounds=[0]*Nsnp
    for k in reversed(list(range(Nsnp))):  # differs from leftbound search algo (I'll put a DIFF label for these lines)

        p,q= k-1,k-1  # DIFF FROM LEFT k+1,k+1
        a,b,d,e = [[] for i in range(4)]
        commonbound=Nsnp #DIFF
        firstDerived=True
        for i in range(Nhap):
            if dcurr[i]<p:  #DIFF
                p=dcurr[i]
            if dcurr[i] <q: #DIFF
                q=dcurr[i]
            if haplos[acurr[i],k]==0:
                a.append(acurr[i])
                d.append(p)
                p=Nsnp-1 #DIFF
            else:
                b.append(acurr[i])
                e.append(q)
                if firstDerived:
                    # When we encounter the first haplotype carrying a derived allele at k
                    # There is no bound to look for yet
                    # because the haplo differs from all previous haplo at position k (previous haplo carries a '0')
                    firstDerived=False
                elif q<commonbound:
                    commonbound=q
                    # commonbound contains the current MINIMAL index such that haplotypes already parsed and carrying '1' at position k are all identical from k to commonbound (included)
                q=Nsnp-1  #DIFF
        acurr=a+b
        dcurr=d+e
        rightbounds[k] = commonbound+1   #DIFF
    # end5=time.time()
    # print(end5-start5)
      
    return rightbounds    

def afibs_durbin_compact(haplos,posOnChrom,counts, afibs=None):
    '''
    Compute afibs length (Theunert 2012) for each allele count >=2
    Algo was adapted from Durbin (PBWT), Bioinformatics 2014


    Args:
    haplos        np.array[Nhap,Nsnp]    haplotype data for one region for Nhap individuals x Nsnp SNPs
    posOnChrom    int array              positions (bp) of each polymorphism relative to its chromosome
    counts        np.array[Nsnp_seg]     number of derived alleles at each position
    afibs         list of list [Nhap]    list of afibs tract length to be updated, if None an empty one will be created


    Returns:
    afibs         list of list [Nhap]     list containing a list of length for each allele count 0...Nhap

    Author: Flora Jay
    '''

    rightbounds =  afibs_durbin_right(haplos)
    afibs =  afibs_durbin_left_length(haplos,posOnChrom,counts,rightbounds, afibs)
    # you could compute more stats
    return afibs

def distrib_afibs(hap, pos, count, durbin_bwt=False):
    """
    Moments for length distributions of AF-IBS as defined by Theunert et al. 2012

    Arguments:
    hap           haplotype data for each segment
    pos           positions of SNP for rach segment
    count         number of derived alleles at each position for each segment
    durbin_bwt      bool                             whether to use algorithm based on durbin ibs algo using Burrows-Wheeler Transform 

    Return:
    mean_sd_afibs   np.array(mean_2,sd_2, mean_3,sd_3, ...)        (mean,sd) of afibs lengths for each category of derived alleles number 2..n
    From FJ
    """
    Nhap=hap.shape[0]
    afibs=[[] for der in range(Nhap+1)]

    if durbin_bwt:
        afibs=afibs_durbin_compact(hap, pos, count, afibs) 
    else:
        print('Set durbin_bwt = True')
        #afibs=afibs_fast(hap==1,pos,count,afibs)   #hap==1 because afibs_fast takes a boolean array as argument, not int

    mean_sd_afibs=np.zeros(shape=(len(afibs)-2)*2)
    # we don't compute afibs values for singletons (does not make sense) nor for fixed derived (because we don't simulated fixed derived, and the ones appearing because of errors added afterwards are pruned)
    i=0
    for der in range(2,len(afibs)):
        if len(afibs[der])>0:
            mean_sd_afibs[i]=np.mean(afibs[der])
            mean_sd_afibs[i+1]=np.std(afibs[der])
        i+=2
    return mean_sd_afibs


def sumstats(ts, simu_settings, sumstats_settings):
    
		
	##### One-way statistics.
	#####     They are defined over a single sample set.
	#####     Computed separately on population 0, population 1 and populations 0 and 1 together.   


	# Proportion of segregating sites over the genome

	S = [ts.segregating_sites(ts.samples(population=0), mode = "site", span_normalise = True),
		ts.segregating_sites(ts.samples(population=1), mode = "site", span_normalise = True),
		ts.segregating_sites(mode = "site", span_normalise = True)]

	# Mean and standard deviation per site Nei's pi (Mean pairwise diversity or Expected Heterozygosity)

	PI_mean = [ts.diversity(ts.samples(population=0), mode = "site", span_normalise = True),
			ts.diversity(ts.samples(population=1), mode = "site", span_normalise = True),
			ts.diversity(mode = "site", span_normalise = True)]

	windows = np.linspace(0, int(ts.sequence_length), int(ts.sequence_length))

	all_PI_0 = ts.diversity(ts.samples(population=0), windows =windows, mode = "site", span_normalise = True)
	all_PI_1 = ts.diversity(ts.samples(population=1), windows =windows, mode = "site", span_normalise = True)
	all_PI = ts.diversity(windows =windows, mode = "site", span_normalise = True)

	PI_std = [np.std(all_PI_0[all_PI_0!= 0]),
			np.std(all_PI_1[all_PI_1!= 0]),
			np.std(all_PI[all_PI!= 0])]

	# Tajima D

	D = [ts.Tajimas_D(ts.samples(population=0), mode = "site"),
		ts.Tajimas_D(ts.samples(population=1), mode = "site"),
		ts.Tajimas_D(mode = "site")]    

	# Mean and standard deviation of haplotypic heterozygosity

	ts0 = ts.simplify(ts.samples(population=0))
	ts1 = ts.simplify(ts.samples(population=1)) 

	pos0 = np.array([variant.site.position for variant in ts0.variants()])
	pos1 = np.array([variant.site.position for variant in ts1.variants()])
	pos = np.array([variant.site.position for variant in ts.variants()])

	hap0 = ts0.genotype_matrix().T
	hap1 = ts1.genotype_matrix().T  
	hap = ts.genotype_matrix().T 
	
	hap0[hap0 > 1] = 1
	hap1[hap1 > 1] = 1
	hap[hap > 1] = 1

	win_size_hh= sumstats_settings['win_size_hh']
	L= simu_settings['L']

	haplo_het_mean0, haplo_het_sd0 = haplo_win(hap0,pos0,win_size_hh,L)
	haplo_het_mean1, haplo_het_sd1 = haplo_win(hap1,pos1,win_size_hh,L)
	haplo_het_mean, haplo_het_sd = haplo_win(hap,pos,win_size_hh,L)

	winHET = [haplo_het_mean0, haplo_het_mean1, haplo_het_mean]
	winHET_std = [haplo_het_sd0, haplo_het_sd1, haplo_het_sd]

	# Site Frequency Spectrum  

	count0 = hap0.sum(axis=0).astype(int)
	count1 = hap1.sum(axis=0).astype(int)
	count = hap.sum(axis=0).astype(int)   

	res_sfs0, res_sfs_dist0 = spatial_histo_fast(pos0, count0, hap0.shape[0])
	res_sfs1, res_sfs_dist1 = spatial_histo_fast(pos1, count1, hap1.shape[0])
	res_sfs, res_sfs_dist = spatial_histo_fast(pos, count, hap.shape[0]) 

	sfs = [res_sfs0, res_sfs1, res_sfs]
	sfs_dist = [res_sfs_dist0, res_sfs_dist1, res_sfs_dist]
 
	sfs_list = res_sfs0.tolist() + res_sfs1.tolist() + res_sfs.tolist()
	sfs_dist_list = res_sfs_dist0.tolist() + res_sfs_dist1.tolist() + res_sfs_dist.tolist() 

	# Linkage disequilibrium

	interval_list = sumstats_settings['interval_list']

	res_LD_mean0, res_LD_var0 = distrib_r2(pos0, hap0, interval_list)
	res_LD_mean1, res_LD_var1 = distrib_r2(pos1, hap1, interval_list)
	res_LD_mean, res_LD_var = distrib_r2(pos, hap, interval_list)

	LD_mean = [res_LD_mean0, res_LD_mean1, res_LD_mean]
	LD_var = [res_LD_var0, res_LD_var1, res_LD_var]  
 
	LD_mean_list = res_LD_mean0.tolist() + res_LD_mean1.tolist() + res_LD_mean.tolist() 
	LD_var_list = res_LD_var0.tolist() + res_LD_var1.tolist() + res_LD_var.tolist() 

	# IBS

	# Quantiles at which IBS distrib is evaluated
	prob_list = sumstats_settings['prob_list']
	# Number of haplotypes to compare (max value should be <=n_haplo)
	size_list = sumstats_settings['size_list']
	# Maximal distance for calculating IBS [usually set to the size of the segment: L]
	dmax = simu_settings['L'] 

	res_IBS_q0 = [ibs_quantiles_from_data(size, pos0, 1, hap0, prob_list, dmax, quantiles=True, moments=False) for size in size_list]
	res_IBS_q1 = [ibs_quantiles_from_data(size, pos1, 1, hap1, prob_list, dmax, quantiles=True, moments=False) for size in size_list]
	res_IBS_q = [ibs_quantiles_from_data(size, pos, 1, hap, prob_list, dmax, quantiles=True, moments=False) for size in size_list]

	IBS_q = [res_IBS_q0, res_IBS_q1, res_IBS_q]
 
	res_IBS_q0_list = [res_IBS_q0[j][k] for j in range(len(size_list)) for k in range(len(prob_list))]
	res_IBS_q1_list = [res_IBS_q1[j][k] for j in range(len(size_list)) for k in range(len(prob_list))] 
	res_IBS_q_list = [res_IBS_q[j][k] for j in range(len(size_list)) for k in range(len(prob_list))] 

	IBS_q_list = res_IBS_q0_list + res_IBS_q1_list + res_IBS_q_list

	# AF-IBS

	res_afibs_bwt0 = distrib_afibs(hap0, pos0, count0, durbin_bwt=True)
	res_afibs_bwt1 = distrib_afibs(hap1, pos1, count1, durbin_bwt=True)
	res_afibs_bwt = distrib_afibs(hap, pos, count, durbin_bwt=True)

	AFIBS = [res_afibs_bwt0, res_afibs_bwt1, res_afibs_bwt] 
	
	AFIBS_list = res_afibs_bwt0.tolist() + res_afibs_bwt1.tolist() + res_afibs_bwt.tolist() 

	##### Multi-way statistics.
	#####     They are defined over several sample sets.
	#####     Considering populations 0 and 1.

	# Divergence,  “average number of differences”, usually referred to as “dxy”;    

	Dxy = [ts.divergence([ts.samples(population=0),ts.samples(population=1)], 
				mode = "site", 
				span_normalise = True)]
	# Fst

	Fst = [ts.Fst([ts.samples(population=0),ts.samples(population=1)], 
				mode = "site", 
				span_normalise = True)]
        

	# Joint allele frequency spectrum

	Jsfs = ts.allele_frequency_spectrum([ts.samples(population=0),ts.samples(population=1)], 
				mode = "site", 
				span_normalise = True,
				polarised = True)

	Jsfs_list =[] 
	Jsfs_list += [Jsfs[i][j] for i in range(len(Jsfs)) for j in range(len(Jsfs))]
 
	sumstats_list = S + PI_mean + PI_std + D + winHET + winHET_std + sfs_list + sfs_dist_list + LD_mean_list + LD_var_list + IBS_q_list + AFIBS_list + Dxy + Fst + Jsfs_list
 
	#return sumstats_list
	return S ,PI_mean , PI_std , D , winHET , winHET_std , sfs_list , sfs_dist_list , LD_mean_list , LD_var_list , IBS_q_list , AFIBS_list , Dxy , Fst , Jsfs_list
 
		
		
		
		











