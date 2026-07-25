import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from multiprocessing import Pool
import multiprocessing
from functools import partial
from collections import Counter
import csv

def omega(l, mQ, r):
    # State Energy
    return math.sqrt(float(l)*float(l)/(r*r) + mQ*mQ)

def gen_omega_list(lmax, mQ, r):
    # Global variable storing a list of all state energies.
    global omega_list
    omega_list = [omega(abs(l), mQ, r) for l in range(0,lmax+1)]
    return omega_list

# --- Helper functions ---
def get_total_occupation_number(state):
    # Returns occupation number for a given state in dictionary format.
    return sum(state.values())


def convert_state_to_list(state, lmax):
    # Returns state held in dictionary to list.
    return [state.get(i, 0) for i in range(-lmax, lmax+1)]


def get_state_energy(state, mQ, r):
    # Returns state energy for state in dictionary format.
    return sum([nl*omega(l, mQ, r) for l, nl in state.items()])


# -----------------------------------------------------------
# -----------------------------------------------------------

def gen_coinDict(coins):
    #Turns coins into a global counting dictionary, which is easier to perform operations on.
    global coinDict
    coinDict = {}
    for key in coins.keys():
        coinDict[key] = [Counter(reversed(val)) for val in coins[key]]
        

def gen_coins(n_max):
    global coins
    # Creates the coins dictionary if it does not already exist.
    if 'coins' not in globals():
        coins = {}
        coins.setdefault(1,[[1]]) 
        
    # If we have already calculated up to n_max, just return the dictionary.
    if n_max in coins: return(coins)
     

    for i in range(max(coins.keys())+1,n_max+1):
        #This step is a little complicated so I'll break it down into the two parts of the sum:
        #   1. The first part adds [1] to the end of every list for the previous element (e.g. 6:[[6],...] -> 7:[...,[6, 1],...])
        #   2. The second part adds 1 to the final element of the previous interation (e.g. 6:[...,[3,2,1],...]->7:[...,[3,2,2],...]) given the following conditions:
        #       a) If the length of the dictionary is 1, just add one (i.e 6:[[6],..] -> 7:[[7],...])
        #       b) If the final two elements of a list are the same, there also exists one where the final two elements differ by two (e.g. 6:[...,[3,3],[4,2],...]).
        #          Without input, this would result in duplicate entries. Therefore, the final two elements must not be equal for this step.

        coins[i] = [cns[:-1]+[cns[-1]+1] for cns in coins[i-1] if (len(cns)==1 or cns[-2]!=cns[-1])] + list(map(lambda x:x+[1], coins[i-1])) 
        #coins[i] = sorted(coins[i], key=len) #Not sure this helps. But re-ordering the coin dictionary after creation in terms of occupation number.
        
    # Creating a counting dictionary version of this as it is easier to perform operations on.
    gen_coinDict(coins)
    return coins


def coin_energies(l,mQ,r):
    #Computes the energies of each possible change.
    return [sum(map(lambda c: omega(c, mQ, r), change)) for change in coins[l]]
            

# -----------------------------------------------------------
# -----------------------------------------------------------



def gen_basis(lmax, e_max, mQ, r):
    iterList = set({0}) # List of states to iterate over.

    #l=0:
    result = [{}] # The empty state.

    for i in range(1, int(math.floor(e_max/mQ)) + 1): # Adding l=0 states up to the maximum possible number allowed before reaching e_max.
        result+=[{0: i}] 
        iterList.add(i)

    #l>0:
    # ix is the running state index which we set to one above the maximum l=0 occupancy.
    # asymList is a set (hashable) containing the indices of asymmetric states.
    # runner keeps track of how many asymmetric states there have previously been to produce the conversion dict.
    ix = int(math.floor(e_max/mQ))+1
    
    
    for l in range(1,lmax+1):
        gen_coins(l) # All ways of making up to lmax with smaller numbers
        gen_coinDict(coins) # Generates a dictionary version
        coinEnergies = coin_energies(l,mQ,r) # Energies for each set of coins.

        # Iterate through positive (ii, ci) and negative (jj, cj) momenta, then merge dictionaries.
        for ii,ci in enumerate(coinDict[l]):
            for jj,cj in enumerate(coinDict[l]):
                if jj>=ii: # Only make one copy of each asymmetric state.

                    #Build a temporary energy for the state so we can 
                    # A) figure out if it exceeds e_max.
                    # B) add l=0 states successively up to e_max.
                    e_temp = e_max - coinEnergies[ii] - coinEnergies[jj]
                    
                    if e_temp > 0: #Fill up l=0 state up to e_max.

                        # Merge positive and negative l states, with total l=0. 

                        negative_cj = {-jjj:cjj for jjj,cjj in reversed(cj.items())} # 'reversed' operation here gives correct l-ordering in the state dictionary. NOTE this is not strictly guaranteed to be maintained so we should adjust this.
                        
                        
                        result+=[ negative_cj | ci ]
                       
                        # Increase state index.
                        iterList.add(ix)
                        ix += 1

                        if ii!=jj: 
                            # Populating the asymList dictionary as states are asymmetric.
                            
                            negative_ci = {-iii:cii for iii,cii in reversed(ci.items())}
                            result += [ negative_ci | cj ]
                        
                            ix += 1



                        # Now adding successively more l=0 states up to total e_max.
                        for i in range(1, int(math.floor(e_temp/mQ)) + 1):

                            # Adding successively more l=0 states up to e_max in the center of the dictionary.
                            result+=[ negative_cj | {0:i} | ci ]
                            
                            iterList.add(ix)
                            ix += 1
                            
                            if ii!=jj: 
                                # Populating the asymList dictionary as states are asymmetric.
 
                                result += [ negative_ci | {0:i} | cj ]
                                ix += 1
                            
                        
                
    return (result, iterList)


#Helper function which calculates the maximum angular momentum pairs which can be produced, still below e_max.
def effective_lmax(lmax, e_max, mQ, r):
    if mQ**2 < e_max**2 / 4:
        return int(min(lmax, math.floor(math.sqrt(max(0, r**2*(e_max**2/4 - mQ**2))))))
    return 0


# Convert the hashable basis to dictionary format.
def basis_to_dict(hashable_basis):
    full_basis_dict = []
    for state in hashable_basis.keys():
        full_basis_dict.append(dict(state)) # dict() wrapper converts tuple to dict.
        
    return full_basis_dict


# -----------------------------------------------------------
# -----------------------------------------------------------


def falling_factorial(x,k):
    # x is the value and k is the number of falling factorial steps to take.
    return math.prod(x - i for i in range(k))

def raising_factorial(x,k):
    # x is the value and k is the number of raising factorial steps to take.
    return math.prod(x + i for i in range(k))



# -----------------------------------------------------------
# -----------------------------------------------------------



def basis_even(basis, iterList):  
    # Collecting only even states, creating a conversion dictionary and a hashable basis.
    # The hashable basis is of the form {tuple representation of state: state index}
    even_basis = []; even_hashable_states = {tuple({}):0}; even_iterlist = set()
    
    basislen = len(basis); runner = 0
    
    for ix in iterList:
        s = basis[ix]
            
        if sum(s.values()) % 2 == 0:
            even_basis += [s]
            even_hashable_states[tuple(s.items())] = runner
            even_iterlist.add(runner)
            runner += 1
            

            if ix<basislen-1 and ix+1 not in iterList: #Might now be able to reduce this code????
                
                a_s = basis[ix+1]
                even_basis += [a_s]
                even_hashable_states[tuple(a_s.items())] = runner
                runner += 1

                    

    return (even_basis, even_hashable_states, even_iterlist)






# -----------------------------------------------------------
# -----------------------------------------------------------

def fill_matrix_pairs(ix, iy, res, ixPaired, iyAbsPaired, iyPresPaired):   
    
    if ixPaired and iyAbsPaired: # ix and iy both in a pair, but iy not the iterated of the pair.
        return ([res, res, res, res], [ix, iy, ix+1, iy-1], [iy, ix, iy-1, ix+1])
            
    elif ixPaired and iyPresPaired: # ix and iy both in a pair, and both the iterated pair.
        return ([res, res, res, res], [ix, iy, ix+1, iy+1], [iy, ix, iy+1, ix+1])

    elif ixPaired: # Just ix in a pair.
        return ([res, res, res, res], [ix, iy, ix+1, iy], [iy, ix, iy, ix+1])
        
    elif iyPresPaired:  # Just iy in a pair.
        return ([res, res, res, res], [ix, iy, ix, iy+1], [iy, ix, iy+1, ix])
        
    else: # No pairs.
        return ([res, res], [ix, iy], [iy, ix])


def populate(ix, state):
        
    omega_list = OMEGA_LIST
    mQ = MQ
    lmaxeff = LMAXEFF
    matrixDim = MATRIXDIM
    iterList = ITERLIST
    hashable_basis = HASHABLE_BASIS
    r = R
    cphi4 = CPHI4
    
        
    H_eff = []; cols = []; rows = []
    
    # --- --- Pre Computation --- ---
    # Pre-Computing some Properties for speed later.
    ixPaired=False
    if ix<matrixDim-1 and ix+1 not in iterList: # If the next term in the sequence is skipped, this is an asymmetric pair.
        ixPaired=True
        
    # --- --- Adding To The Diagonal (H_0) --- ---

    # Add energy to the diagonal.

    if ixPaired:
        # If the state is asymmetric, buff H_eff with an additional entry on the next diagonal.
        res = get_state_energy(state, mQ, r)
        H_eff.extend([res,res]); rows.extend([ix,ix+1]); cols.extend([ix,ix+1])
    else:
        # Otherwise just add to the current diagonal.
        res = get_state_energy(state, mQ, r)
        H_eff.extend([res]); rows.extend([ix]); cols.extend([ix])
        

    # --- --- Adding H_4 Operations --- ---

    # All options here first begin removing two occupants from the state so beginning with this.
    for l,nl in state.items():
        if nl == 0: continue
        
        l_base = np.sqrt(nl) * 1/np.sqrt(omega_list[abs(l)])

        # Modify the state
        state[l] -= 1

        for ll,nll in state.items():
            if nll == 0: continue
            
            # --- Preparing --- 
            if ll>l: break # This clause automatically terminates anything which would double-count.
            if abs(l+ll)>lmaxeff:continue # Avoid encountering unreachable states with momentum above lmaxeff
            
            ll_base = l_base * np.sqrt(nll) * 1/np.sqrt(omega_list[abs(ll)])
            if ll == l: ll_base *= (1/2)
            
            # Modify the state again
            state[ll] -= 1

            # --- --- Note for the following operations, multiple additions will likely occur to the same state. coo_matrix then sums all entries in the same location.

            # ------------------------------------------
                # 2 raising 2 lowering operations.
            # ------------------------------------------
            # --- Zero Difference in Occupation Number ---
            

            res = cphi4 * ll_base ** 2
            
            # Apply the Diagonal with the \phi^4 operator.
            if ixPaired:
                # If asymmetric, add to _ix diagonal and _ix+1 diagonal.
                H_eff.extend([res, res]); rows.extend([ix,ix+1]); cols.extend([ix,ix+1])
            else:
                # If not asymmetric, only add to _ix diagonal.
                H_eff.extend([res]); rows.extend([ix]); cols.extend([ix])

            

            
            if abs(ll-l)>1: # Note this again avoids double-counting. I.e. if l=-ll, l_total removed is zero. Either the add_l and add_ll are equal (in which case we recover the case above), or they differ (in which case another iteration of the loop will catch this condition).
                

                # We iterate through each addition without double-counting by (recall we have set l>ll):
                #  A) reducing the added momenta from l by one, 
                #  B) increasing the added momenta from ll by one. 
                # This ensures the total added angular momentum is still identical to before, and we do not double-count states.

                # This clause calculates the maximum number of times we perform the procedure before we double-count.
                iteration = math.ceil((abs(ll-l)+1)/2) if -ll!=l else abs(ll)+1 
                
                
                for i in reversed(range(1, iteration)):
                    # As mentioned, adding to ll and removing from l sequentially.
                    stA = ll+i; stB = l-i
                    
                    # Creating a new dictionary representing the target state.
                    if stA not in state.keys(): state[stA] = 1
                    else: state[stA] += 1
                    if stB not in state.keys(): state[stB] = 1
                    else: state[stB] += 1
                    final_state = tuple(sorted([(a,b) for a,b in state.items() if b!= 0])) # Re-sort the target state into the correct ordering, to use the hash table.
                    
                    
                    
                    nstA = state[stA]; estA = omega_list[abs(stA)]
                    # Calculating the transition matrix element for adding stA and stB back on, following Eq. 3.10 in the draft.
                    if stA == stB:
                        res = (cphi4) * ll_base * np.sqrt((nstA-1)*(nstA)) / (2 * estA)
                    else:
                        nstB = state[stB]; estB = omega_list[abs(stB)]
                        res = (cphi4) * ll_base * np.sqrt((nstA)*(nstB) / (estA * estB))
                    
                    
                    # Look up the new target state in the hashable_basis table to find the index of the new state.
                    iy = hashable_basis.get(final_state)

                    # Adds to H_eff both the transition matrix element for removing and adding (basically a transpose).
                    _H_eff, _cols, _rows = fill_matrix_pairs(ix, iy, res, ixPaired, iy not in iterList, iy+1 not in iterList and iy+1<matrixDim)
                    H_eff.extend(_H_eff); cols.extend(_cols); rows.extend(_rows)
                

                    # Return state back to it's original form.
                    state[stA] -= 1
                    if state[stA] == 0 and stA not in {l,ll}:
                        del state[stA]
                    state[stB] -= 1
                    if state[stB] == 0 and stB not in {l,ll}:
                        del state[stB]


            
            for lll,nlll in state.items():
                if nlll == 0: continue
                if lll>ll: break # Again this clause automatically terminates anything which would double-count.
                        

                    
                state[lll]-=1
                
                # To maintain zero total angular momentum, we will add the state add_l and attempt to remove the state sub_l.
                add_l = (l+ll+lll)
                sub_l = -(l+ll+lll)
                
                lll_base = ll_base * np.sqrt(nlll/(omega_list[abs(add_l)] * omega_list[abs(lll)]))
                
                if lll == ll == l: lll_base *= (1/3)
                elif lll == ll or lll == l: lll_base *= (1/2)
                
                # ------------------------------------------
                    # 3 lowering 1 raising operations.
                # ------------------------------------------
                
                # Modify dictionary representation of the new state.
                if add_l in state.keys(): # If this particular angular momentum exists in the dictionary we just add it on.
                    state[add_l]+=1 
                    final_state = tuple((a,b) for a,b in state.items() if b!= 0)
                else: # If this particular angular momentum does not exist in the dictionary we add it and must now re-sort the dictionary to use the hashable basis table.
                    state[add_l]=1
                    final_state = tuple(sorted([(a,b) for a,b in state.items() if b!= 0]))
                
                # Calculation of the transition matrix entry follows draft Eq. 3.10. First line is the contribution from removing the 3 states. The second line is from adding add_l.
                
                res = (cphi4) * np.sqrt(state[add_l]) * lll_base
                
                # Look up the new target state in the hashable_basis table to find the index of the new state.
                iy = hashable_basis.get(final_state)
                    
                _H_eff, _cols, _rows = fill_matrix_pairs(ix, iy, res, ixPaired, iy not in iterList, iy+1 not in iterList and iy+1<matrixDim)
                H_eff.extend(_H_eff); cols.extend(_cols); rows.extend(_rows)
                
                


                # Return new_dict back to it's original form.
                state[add_l] -= 1
                if state[add_l] == 0 and add_l not in {l, ll, lll}:
                    del state[add_l]

                # ------------------------------------------
                    # 4 lowering operations.
                # ------------------------------------------

                if sub_l<=lll and sub_l in state.keys() and state[sub_l] != 0:  # This allows us to avoid double-counting.

                    if sub_l == lll == ll == l: lll_base *= (1/4)
                    elif sub_l == lll == ll or sub_l == lll == ll or sub_l == ll == l: lll_base *= (1/3)
                    elif sub_l == lll or sub_l == ll or sub_l == l: lll_base *= (1/2)

                    res = (cphi4) * lll_base * np.sqrt(state[sub_l])
                    
                    state[sub_l]-=1
                            
                    
                    final_state = tuple((a,b) for a,b in state.items() if b!= 0)
                    # Look up the new target state in the hashable_basis table to find the index of the new state.
                    iy = hashable_basis.get(final_state)
                    

                    _H_eff, _cols, _rows = fill_matrix_pairs(ix, iy, res, ixPaired, iy not in iterList, iy+1 not in iterList and iy+1<matrixDim)
                    H_eff.extend(_H_eff); cols.extend(_cols); rows.extend(_rows)
                            

                    # Return new_dict back to it's original form.
                    state[sub_l] += 1
                
                # Return new_dict back to it's original form.
                state[lll] += 1

            # Return new_dict back to it's original form.
            state[ll] += 1


        # Return new_dict back to it's original form.
        state[l] += 1
        
    return np.array([H_eff, rows, cols])



HASHABLE_BASIS = None
OMEGA_LIST = None
CPHI4 = None
MQ= None
LMAXEFF = None
ITERLIST = None
R = None
MATRIXDIM = None

def init_worker(hashable_basis, omega_list, cphi4, mQ, lmaxeff, iterList, r, matrixDim):
    global HASHABLE_BASIS, OMEGA_LIST, CPHI4, MQ, LMAXEFF, ITERLIST, R, MATRIXDIM
    HASHABLE_BASIS = hashable_basis
    OMEGA_LIST = omega_list
    CPHI4 = cphi4
    MQ = mQ
    LMAXEFF = lmaxeff
    ITERLIST = iterList
    R = r
    MATRIXDIM = matrixDim

def transition_matrix_phi4(mQ, g, r, lmaxeff, basis_dict, hashable_basis, iterList):

    tasks = zip(iterList, basis_dict)

    cphi4=g*(1/(8*math.pi))*(1/r)
    matrixDim = len(hashable_basis)

    with Pool(
        processes=4,
        initializer=init_worker,
        initargs=(hashable_basis, omega_list, cphi4, mQ, lmaxeff, iterList, r, matrixDim),
    ) as pool:
        results = pool.starmap(populate, tasks, chunksize=500)
        
    results = np.concatenate(results, axis=1)
    
    return sp.coo_matrix((results[0,:],(results[1,:], results[2,:])), shape=(matrixDim,matrixDim))



if __name__ == "__main__":

    
    # --- Hamiltonian Parameters for Test ---
    g = 4*np.pi; mV = 1.; mQ = 1.; r = 1.; e_max=10; lmaxeff = effective_lmax(e_max,e_max,mQ,r)

    # --- Generate the Basis ---
    dict_basis, hashable_basis, iterList = basis_even(*gen_basis(lmaxeff,e_max,mQ,r)) #Select Even Basis States Only.
    
    gen_omega_list(lmaxeff, mQ, r)
    # --- Run the transition matrix element calculation in phi^4 ---
    H_eff_coo=transition_matrix_phi4(mQ, g, r, lmaxeff, dict_basis, hashable_basis, iterList)

    # --- Prepare the Plot ---

    # 1. Convert the reduced basis into the buffed version with both anti-symmetric pairs.
    full_basis=dict_basis


    # 2. Compute properties.
    occupation_numbers = np.array([get_total_occupation_number(state) for state in full_basis]) # State Occupation Number for each in basis.
    full_state_energies = np.array([get_state_energy(state, mQ, r) for state in full_basis]) ## State Energies for each in basis.


    # 3. Sort the basis using lexsort, primarily by occupation number, secondly by energy.
    double_sorted_indices = np.lexsort((full_state_energies, occupation_numbers)) 
    sorted_full_basis = [full_basis[i] for i in double_sorted_indices]
    H_sorted = H_eff_coo.toarray()[double_sorted_indices, :][:, double_sorted_indices]


    # 4. Identify unique occupation values and their starting indices
    # (Using the sorted array guarantees these indices mark the boundaries)
    unique_occ, boundaries_vec = np.unique(occupation_numbers[double_sorted_indices], return_index=True)

    # 5. Map the boundaries to a dictionary
    boundaries = dict(zip(unique_occ, boundaries_vec))


    # --- --- Plot --- ---

    # --- State labels ---
    verbose_labels = []
    #for state in sorted_full_basis:
    for state in full_basis:
        verbose_labels.append("{"+", ".join(f"{key}: {value}" for key, value in sorted(state.items()) if value !=0)+"}") # Such that they 'look' like a dictionary of states on the plot.

    # --- Boundaries between occupation numbers for the plot ---
    state_vecs = np.array([convert_state_to_list(state, lmaxeff) for state in sorted_full_basis])
    boundaries = np.where(np.diff(np.sort(np.sum(state_vecs, axis=1))))[0] # List of places where state occupation number changes.
    boundaries[0] = 0 # Fixing the zero entry.

    # --- Plotting the Heatmap Matrix ---
    plt.figure(figsize=(12, 12))

    zero_mask = (H_sorted == 0.0) # So that we don't have an overcrowded plot.\
    zero_mask = (H_eff_coo.toarray() == 0.0)
    ax = sns.heatmap(
        H_eff_coo.toarray(), 
        annot=True, 
        fmt=".2f", 
        cmap="magma", 
        linewidths=0.8,
        xticklabels=verbose_labels, 
        yticklabels=verbose_labels,
        mask=zero_mask,
        cbar_kws={'label': 'Matrix Element Magnitude $\\langle \\psi_{out} | H | \\psi_{in} \\rangle$'}
    )


    # Drawing separation dividers between states of differing occupation number.
    for b in boundaries:
        ax.axvline(x=b+1, linewidth=2, color="grey")
        ax.axhline(y=b+1, linewidth=2, color="grey")


    ax.xaxis.tick_top() # Layout of x-axis.
    ax.xaxis.set_label_position('top')
    ax.set_facecolor('#eaeaea') # Soft gray for zero elements
    plt.xticks(rotation=45, ha='left', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.savefig(f'phi2_matrix_creation_r{r}mQ{mQ}mV{mV}emax{e_max}.pdf',format='pdf')
    plt.tight_layout()
    plt.show()

    
    