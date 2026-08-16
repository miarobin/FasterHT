import math
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from collections import Counter

def omega(l, mQ, r):
    # State Energy
    return math.sqrt(float(l)*float(l)/(r*r) + mQ*mQ)

def gen_omega_list(lmax, mQ, r):
    # Global variable storing a list of all state energies.
    global omega_list
    omega_list = [omega(abs(l), mQ, r) for l in range(0,lmax+1)]

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

def gen_parition_counter(partitions):
    #Turns partitions into a global counting dictionary, which is easier to perform operations on.
    global PARTITION_COUNTER
    PARTITION_COUNTER = {}
    for key in partitions.keys():
        PARTITION_COUNTER[key] = [Counter(reversed(val)) for val in partitions[key]]
        

def gen_parts(n_max):
    global PARTITIONS
    # Creates the PARTITIONS dictionary if it does not already exist.
    if 'PARTITIONS' not in globals():
        PARTITIONS = {}
        PARTITIONS.setdefault(1,[[1]]) 
        
    # If we have already calculated up to n_max, just return the dictionary.
    if n_max in PARTITIONS: return(PARTITIONS)
     

    for i in range(max(PARTITIONS.keys())+1,n_max+1):
        #This step is a little complicated so I'll break it down into the two parts of the sum:
        #   1. The first part adds [1] to the end of every list for the previous element (e.g. 6:[[6],...] -> 7:[...,[6, 1],...])
        #   2. The second part adds 1 to the final element of the previous interation (e.g. 6:[...,[3,2,1],...]->7:[...,[3,2,2],...]) given the following conditions:
        #       a) If the length of the dictionary is 1, just add one (i.e 6:[[6],..] -> 7:[[7],...])
        #       b) If the final two elements of a list are the same, there also exists one where the final two elements differ by two (e.g. 6:[...,[3,3],[4,2],...]).
        #          Without input, this would result in duplicate entries. Therefore, the final two elements must not be equal for this step.

        PARTITIONS[i] = [cns[:-1]+[cns[-1]+1] for cns in PARTITIONS[i-1] if (len(cns)==1 or cns[-2]!=cns[-1])] + list(map(lambda x:x+[1], PARTITIONS[i-1])) 
        
    # Creating a counting dictionary version of this as it is easier to perform operations on.
    gen_parition_counter(PARTITIONS)
    return PARTITIONS


def partition_energies(l,mQ,r):
    #Computes the energies of each possible partition for the state l.
    return [sum(map(lambda c: omega(c, mQ, r), change)) for change in PARTITIONS[l]]
            

# -----------------------------------------------------------
# -----------------------------------------------------------



def gen_basis(lmax, e_max, mQ, r):
    iterSet = set({0}) # Set of states to iterate over when generating the transition matrix.

    #l=0:
    result = [{}] # The empty state.

    for i in range(1, int(math.floor(e_max/mQ)) + 1): # Adding l=0 states up to the maximum possible number allowed before reaching e_max.
        result+=[{0: i}] 
        iterSet.add(i)

    #l>0:
    # ix is the running state index which we set to one above the maximum l=0 occupancy.
    ix = int(math.floor(e_max/mQ))+1
    
    
    for l in range(1,lmax+1):
        gen_parts(l) # All ways of making up to lmax with smaller numbers
        gen_parition_counter(PARTITIONS) # Generates a dictionary version
        partEnergies = partition_energies(l,mQ,r) # Energies for each set of coins.

        # Iterate through positive (ii, ci) and negative (jj, cj) momenta, then merge dictionaries.
        for ii,ci in enumerate(PARTITION_COUNTER[l]):
            for jj,cj in enumerate(PARTITION_COUNTER[l]):
                if jj>=ii: # Only make one copy of each asymmetric state.

                    #Build a temporary energy for the state so we can 
                    # A) figure out if it exceeds e_max.
                    # B) add l=0 states successively up to e_max.
                    e_temp = e_max - partEnergies[ii] - partEnergies[jj]
                    
                    if e_temp > 0: #Fill up l=0 state up to e_max.

                        # Merge positive and negative l states, with total l=0. 
                        negative_cj = {-jjj:cjj for jjj,cjj in reversed(cj.items())} # 'reversed' operation here gives correct l-ordering in the state dictionary. NOTE this is not strictly guaranteed to be maintained.
                        
                        
                        result+=[ negative_cj | ci ]
                       
                        # Add to the iterSet and increase state index.
                        iterSet.add(ix)
                        ix += 1

                        if ii!=jj: 
                            # Merge the Z2 reversed dictionaries.
                            negative_ci = {-iii:cii for iii,cii in reversed(ci.items())}
                            result += [ negative_ci | cj ]
                        
                            # Increase state index but do not add to iterSet.
                            ix += 1



                        # Now adding successively more l=0 states up to total e_max.
                        for i in range(1, int(math.floor(e_temp/mQ)) + 1):

                            # Adding successively more l=0 states up to e_max in the center of the dictionary.
                            result+=[ negative_cj | {0:i} | ci ]
                            
                            iterSet.add(ix)
                            ix += 1
                            
                            if ii!=jj: 
                                # Merge the Z2 reversed dictionaries.
                                result += [ negative_ci | {0:i} | cj ]
                                ix += 1
                            
                        
                
    return (result, iterSet)


# -----------------------------------------------------------
# -----------------------------------------------------------


#Helper function which calculates the maximum angular momentum pairs which can be produced, still below e_max.
def effective_lmax(lmax, e_max, mQ, r):
    if mQ**2 < e_max**2 / 4:
        return int(min(lmax, math.floor(math.sqrt(max(0, r**2*(e_max**2/4 - mQ**2))))))
    return 0

def falling_factorial(x,k):
    # x is the value and k is the number of falling factorial steps to take.
    return math.prod(x - i for i in range(k))

def raising_factorial(x,k):
    # x is the value and k is the number of raising factorial steps to take.
    return math.prod(x + i for i in range(k))



# -----------------------------------------------------------
# -----------------------------------------------------------

def basis_even(basis, iterSet):  
    # Collecting only even states, creating a hashable basis to locate the position of a particular state in the basis.
    # The hashable basis is of the form {tuple representation of state: state index}
    
    even_basis = []; even_hashable_states = {tuple({}):0}; even_iterset = set()
    
    basislen = len(basis); runner = 0 # runner to keep track of the basis index.
    

    for ix in iterSet: # Iterate through the iterSet list of the minimum necessary states to compute.
        s = basis[ix] # Grabs the corresponding dictionary representation of the state
            
        if sum(s.values()) % 2 == 0: # Confirm occupancy number is even.
            # If so add to the necessary data structures.
            even_basis += [s] 
            even_hashable_states[tuple(s.items())] = runner
            even_iterset.add(runner)
            # Increase state index by one.
            runner += 1
            
            
            if ix<basislen-1 and ix+1 not in iterSet: # Confirm if state is part of Z2 antisymmetric pair.
                # If so, add that to (hashable) basis. Do not add to the iterset.
                a_s = basis[ix+1]
                even_basis += [a_s]
                even_hashable_states[tuple(a_s.items())] = runner
                # Increase state index by one.
                runner += 1


    return (even_basis, even_hashable_states, even_iterset)


def basis_odd(basis, iterSet):  
    # Collecting only odd states, creating a hashable basis to locate the position of a particular state in the basis.
    # The hashable basis is of the form {tuple representation of state: state index}
    odd_basis = []; odd_hashable_states = {tuple({}):0}; odd_iterset = set()
    
    basislen = len(basis); runner = 0 # runner to keep track of the basis index.
    
    for ix in iterSet: # Iterate through the iterSet list of the minimum necessary states to compute.
        s = basis[ix] # Grabs the corresponding dictionary representation of the state
            
        if sum(s.values()) % 2 != 0: # Confirm occupancy number is odd.
            # If so add to the necessary data structures.
            odd_basis += [s]
            odd_hashable_states[tuple(s.items())] = runner
            odd_iterset.add(runner)
            # Increase state index by one.
            runner += 1

            if ix<basislen-1 and ix+1 not in iterSet: # Confirm if state is part of Z2 antisymmetric pair.
                # If so, add that to (hashable) basis. Do not add to the iterset.
                a_s = basis[ix+1]
                odd_basis += [a_s]
                odd_hashable_states[tuple(a_s.items())] = runner
                # Increase state index by one.
                runner += 1


    return (odd_basis, odd_hashable_states, odd_iterset)


def basis_all(basis, iterSet):  
    # Collecting all states, creating a hashable basis to locate the position of a particular state in the basis.
    # The hashable basis is of the form {tuple representation of state: state index}
    all_basis = []; all_hashable_states = {tuple({}):0}; all_iterset = set()
    
    basislen = len(basis); runner = 0 # runner to keep track of the basis index.
    
    for ix in iterSet:
        # Grabs the corresponding dictionary representation of the state
        s = basis[ix]
        
        # Add state to the necessary data structures.
        all_basis += [s]
        all_hashable_states[tuple(s.items())] = runner
        all_iterset.add(runner)
        # Increase state index by one.
        runner += 1

        if ix<basislen-1 and ix+1 not in iterSet: # Confirm if state is part of Z2 antisymmetric pair.
            # If so, add that to (hashable) basis. Do not add to the iterset.
            a_s = basis[ix+1]
            all_basis += [a_s]
            all_hashable_states[tuple(a_s.items())] = runner
            # Increase state index by one.
            runner += 1

    return (all_basis, all_hashable_states, all_iterset)


# -----------------------------------------------------------
# -----------------------------------------------------------

def fill_matrix_pairs(ix, iy, res, ixPaired, iyAbsPaired, iyPresPaired):   
    # A function to fill Z2 antisymmetric pairs into the transition matrix consistently.
    #
    # iy and ix are the results from transition_matrix_phi2 and transition_matrix_phi4's operations and:
    #
    #   ixPaired: Bool, if the ix is part of a Z2 symmetric pair. Checked with `ix<matrixDim-1 and ix+1 not in iterSet`
    #   iyAbsPaired : Bool, the iy is the absent of the pair from iterSet. Checked with `iy not in iterSet`
    #   iyPresPaired : Bool, the iy is the present of the pair from iterSet. Checked with `iy+1 not in iterSet and iy+1<matrixDim`
    #   (If both are False, iy is not part of a Z2 symmetric pair. Note both cannot be simultaneously Tru.e)
    #
    
    if ixPaired and iyAbsPaired: # ix and iy both in a pair, but iy not the iterated of the pair.
        return ([res, res, res, res], [ix, iy, ix+1, iy-1], [iy, ix, iy-1, ix+1])
            
    elif ixPaired and iyPresPaired: # ix and iy both in a pair, and both the iterated pair.
        return ([res, res, res, res], [ix, iy, ix+1, iy+1], [iy, ix, iy+1, ix+1])

    elif ixPaired: # Just ix in a pair.
        return ([res, res, res, res], [ix, iy, ix+1, iy], [iy, ix, iy, ix+1])
        
    else: # No pairs.
        return ([res, res], [ix, iy], [iy, ix])
        
        


def populate(ix, state):
    
    # Pulling the global variables.
    omega_list = OMEGA_LIST
    mQ = MQ
    lmaxeff = LMAXEFF
    matrixDim = MATRIXDIM
    iterSet = ITERSET
    hashable_basis = HASHABLE_BASIS
    r = R
    cphi4 = CPHI4
    
        
    H_eff = []; cols = []; rows = []
    
    # -------------------------------
    # ------- Pre Computation -------
    # -------------------------------
    # Pre-Computing some Properties for speed later.
    ixPaired=False
    if ix<matrixDim-1 and ix+1 not in iterSet: # If the next term in the sequence is skipped, this is an asymmetric pair.
        ixPaired=True

    # -------------------------------
    #  Adding To The Diagonal (H_0) -
    # -------------------------------

    # Add energy to the diagonal:
    if ixPaired:
        # If the state is Z2 antisymmetric, buff H_eff with an additional entry on the next diagonal.
        res = get_state_energy(state, mQ, r)
        H_eff.extend([res,res]); rows.extend([ix,ix+1]); cols.extend([ix,ix+1])
    else:
        # Otherwise just add to the current diagonal.
        res = get_state_energy(state, mQ, r)
        H_eff.extend([res]); rows.extend([ix]); cols.extend([ix])


    # -------------------------------
    # ---- Adding H_4 Operations ----
    # -------------------------------

    # All operations contain at least two annihilation operators.

    # --- Annihilation of l ---
    for l,nl in state.items():
        if nl == 0: continue # Confirming there is an occupier in the state.
        
        # Computation of base factor for removing momentum l.
        l_base = math.sqrt(nl/omega_list[abs(l)])

        # Modify current state.
        state[l] -= 1
        # NOTE if occupancy drops to zero, do not remove from dictionary. It helps to preserve ordering.

        # --- Annihilation of ll ---
        for ll,nll in state.items():
            if nll == 0: continue # Confirming there is an occupier in the state.
            
            # --- --- --- --- --- --- --- --- --- --- --- --- --
            # --- --- --- --- 0. Preparations --- --- --- --- --
            # --- --- --- --- --- --- --- --- --- --- --- --- --
            
            if ll>l: break # This clause automatically terminates anything which would double-count.
            if abs(l+ll)>lmaxeff:continue # Avoid encountering unreachable states with momentum above lmaxeff
            
            # Computation of the base factor for removing momenta l and ll.
            ll_base = l_base * math.sqrt(nll/omega_list[abs(ll)])
            # Fix factor if l and ll identical.
            if ll == l: ll_base *= (1/2)
            
            # Modify current state again.
            state[ll] -= 1


            # --- --- --- --- --- --- --- --- --- --- --- --- --
            # -- 1. Remove two states and add two states back --
            # ---- Corresponds to the sixth row of Table 2. ----
            # --- --- --- --- --- --- --- --- --- --- --- --- --
            

            # --- Adding and removing the same two states (i.e. l and ll) ---
            # Transition Amplitude
            res = cphi4 * ll_base ** 2
            
            # Apply the Diagonal with the \phi^4 operator.
            if ixPaired:
                # If the state is Z2 antisymmetric, buff H_eff with an additional entry on the next diagonal.
                H_eff.extend([res, res]); rows.extend([ix,ix+1]); cols.extend([ix,ix+1])
            else:
                # Otherwise just add to the current diagonal.
                H_eff.extend([res]); rows.extend([ix]); cols.extend([ix])

            # --- --- --- --- --- --- --- --- --- --- --- --- --
            # - 2. Remove two and add back two different states-
            # Corresponds to the fourth and fifth row of Table 2.
            # --- --- --- --- --- --- --- --- --- --- --- --- --

            if abs(ll-l)>1: # Note this again avoids double-counting. 
                # I.e. if l=-ll, l_total removed is zero. Either the add_l and add_ll are equal (in which case we recover the case above), or they differ (in which case another iteration of the loop will catch this condition).
                

                # We iterate through each addition without double-counting by (recall we have set l>ll):
                #  A) reducing the added momenta from l by one, 
                #  B) increasing the added momenta from ll by one. 
                # This ensures the total added angular momentum is still identical to before, and we do not double-count states.

                # This clause calculates the maximum number of times we perform the procedure before we double-count.
                iteration = math.ceil((abs(ll-l)+1)/2) if -ll!=l else abs(ll)+1 
                
                
                for i in reversed(range(1, iteration)):
                    # As mentioned, adding to ll and removing from l sequentially.
                    stA = ll+i; stB = l-i
                    
                    # Updating the current state with the added operations.
                    # --- Creation of stA ---
                    if stA not in state.keys(): state[stA] = 1
                    else: state[stA] += 1
                    # --- Creation of stB ---
                    if stB not in state.keys(): state[stB] = 1
                    else: state[stB] += 1
                    
                    # Re-sort the target state into the correct ordering, and convert to tuple to use the hashtable.
                    final_state = tuple(sorted([(a,b) for a,b in state.items() if b!= 0])) 
                    
                    
                    nstA = state[stA]; estA = omega_list[abs(stA)]
                    # Calculating the transition matrix element for adding stA and stB back on.
                    if stA == stB:
                        res = (cphi4) * ll_base * math.sqrt((nstA-1)*(nstA)) / (2 * estA)
                    else:
                        nstB = state[stB]; estB = omega_list[abs(stB)]
                        res = (cphi4) * ll_base * math.sqrt((nstA)*(nstB) / (estA * estB))
                    
                    
                    # Look up the new target state in the hashable_basis table to find the index of the new state.
                    iy = hashable_basis.get(final_state)

                    # Adds to H_eff both the transition matrix element for removing and adding (basically a transpose).
                    _H_eff, _cols, _rows = fill_matrix_pairs(ix, iy, res, ixPaired, iy not in iterSet, iy+1 not in iterSet and iy+1<matrixDim)
                    H_eff.extend(_H_eff); cols.extend(_cols); rows.extend(_rows)
                

                    # Return state back to it's previous form.
                    state[stA] -= 1
                    if state[stA] == 0 and stA not in {l,ll}:
                        del state[stA]
                    state[stB] -= 1
                    if state[stB] == 0 and stB not in {l,ll}:
                        del state[stB]


            # --- Annihilation of lll ---
            for lll,nlll in state.items():
                if nlll == 0: continue # Confirming we can do this.
                if lll>ll: break # Again this clause automatically terminates anything which would double-count.
                        
                # Modify current state again.
                state[lll]-=1
                
                # To maintain zero total angular momentum, we will add the state add_l and attempt to remove the state sub_l.
                add_l = (l+ll+lll)
                sub_l = -(l+ll+lll)
                
                # Computation of the base factor for removing momenta l, ll and lll. 
                # NOTE This is also scaled by 1/sqrt(energy) of the add_l/sub_l state, as it is the same for both following operations.
                lll_base = ll_base * math.sqrt(nlll/(omega_list[abs(add_l)] * omega_list[abs(lll)]))
                
                # Fixing the symmetry factor for repeated operations.
                if lll == ll == l: lll_base *= (1/3)
                elif lll == ll or lll == l: lll_base *= (1/2)
                

                # --- --- --- --- --- --- --- --- --- --- --- --- --
                # -- 3. Remove three states and add one state back -
                # ---- Corresponds to the second row of Table 2. ---
                # --- --- --- --- --- --- --- --- --- --- --- --- --
                
                # --- Creation of add_l ---
                if add_l in state.keys(): # If this particular angular momentum exists in the dictionary we just add it on.
                    state[add_l]+=1 
                    # Tuple representation to use the hashable_basis. Remove non-occupied momenta.
                    final_state = tuple((a,b) for a,b in state.items() if b!= 0)
                else: # If this particular angular momentum does not exist in the dictionary we add it and must now re-sort the dictionary to use the hashable basis table.
                    state[add_l]=1
                    # Tuple representation to use the hashable_basis. Remove non-occupied momenta and sort correctly.
                    final_state = tuple(sorted([(a,b) for a,b in state.items() if b!= 0]))
                
                # Calculation of the transition matrix entry.
                res = (cphi4) * math.sqrt(state[add_l]) * lll_base
                
                # Look up the new target state in the hashable_basis table to find the index of the new state.
                iy = hashable_basis.get(final_state)
                
                # Adds to H_eff both the transition matrix element for removing and adding (basically a transpose), as well as Z2 antisymmetric pairs to the correct location.
                _H_eff, _cols, _rows = fill_matrix_pairs(ix, iy, res, ixPaired, iy not in iterSet, iy+1 not in iterSet and iy+1<matrixDim)
                H_eff.extend(_H_eff); cols.extend(_cols); rows.extend(_rows)
                
                
                # Return current state back to it's original form.
                state[add_l] -= 1
                if state[add_l] == 0 and add_l not in {l, ll, lll}:
                    del state[add_l]


                # --- --- --- --- --- --- --- --- --- --- --- --- --
                # --- --- ---- 4. Remove four states --- --- --- ---
                # --- - Corresponds to the first row of Table 2. ---
                # --- --- --- --- --- --- --- --- --- --- --- --- --

                # --- Annihilation of sub_l ---
                if sub_l<=lll and sub_l in state.keys() and state[sub_l] != 0:  # This allows us to avoid double-counting, and confirms there is an occupant for us to remove.

                    # Fixing the symmetry factor for repeated operations.
                    if sub_l == lll == ll == l: lll_base *= (1/4)
                    elif sub_l == lll == ll or sub_l == lll == ll or sub_l == ll == l: lll_base *= (1/3)
                    elif sub_l == lll or sub_l == ll or sub_l == l: lll_base *= (1/2)

                    # Calculation of the transition matrix entry.
                    res = (cphi4) * lll_base * math.sqrt(state[sub_l])
                    
                    # Modify current state again.
                    state[sub_l]-=1
                            
                    # Tuple representation to use the hashable_basis. Remove non-occupied momenta.
                    final_state = tuple((a,b) for a,b in state.items() if b!= 0)
                    # Look up the new target state in the hashable_basis table to find the index of the new state.
                    iy = hashable_basis.get(final_state)
                    

                    # Adds to H_eff both the transition matrix element for removing and adding (basically a transpose), as well as Z2 antisymmetric pairs to the correct location.
                    _H_eff, _cols, _rows = fill_matrix_pairs(ix, iy, res, ixPaired, iy not in iterSet, iy+1 not in iterSet and iy+1<matrixDim)
                    H_eff.extend(_H_eff); cols.extend(_cols); rows.extend(_rows)
                            

                    # Return current state back to it's original form.
                    state[sub_l] += 1
                
                # Return current state back to it's original form.
                state[lll] += 1

            # Return current state back to it's original form.
            state[ll] += 1


        # Return current state back to it's original form.
        state[l] += 1

        
    return np.array([H_eff, rows, cols])


# Ensures the large list of information is passed properly.
HASHABLE_BASIS = None
OMEGA_LIST = None
CPHI4 = None
MQ = None
LMAXEFF = None
ITERSET = None
R = None
MATRIXDIM = None

def init_worker(hashable_basis, omega_list, cphi4, mQ, lmaxeff, iterSet, r, matrixDim):
    # Helper function so that each worker only needs to copy this information across once (as it has a large overhead).
    global HASHABLE_BASIS, OMEGA_LIST, CPHI4, MQ, LMAXEFF, ITERSET, R, MATRIXDIM
    HASHABLE_BASIS = hashable_basis
    OMEGA_LIST = omega_list
    CPHI4 = cphi4
    MQ = mQ
    LMAXEFF = lmaxeff
    ITERSET = iterSet
    R = r
    MATRIXDIM = matrixDim

def transition_matrix_phi4(mQ, r, cphi4, lmaxeff, basis_dict, hashable_basis, iterSet):
    # Prepare indices and states (these are all Z2 symmetric states and one of Z2 antisymmetric pairs) to pass to the pool.
    tasks = ((i, basis_dict[i]) for i in iterSet)

    # Length of the basis.
    matrixDim = len(hashable_basis)

    # Construct the pool. Pass all necessary arguments to the workers as global variables.
    with Pool(
        processes=4, 
        initializer=init_worker,
        initargs=(hashable_basis, omega_list, cphi4, mQ, lmaxeff, iterSet, r, matrixDim),
    ) as pool:
        # Pass the tasks to the 4 cores.
        results = pool.starmap(populate, tasks, chunksize=500)
        
    # Flatten results into three columns: entry values, column and row locations.
    results = np.concatenate(results, axis=1)
    # Turns this into a sparse matrix.
    return sp.coo_matrix((results[0,:],(results[1,:], results[2,:])), shape=(matrixDim,matrixDim))



if __name__ == "__main__":


    # ---------------------------------
    # ---- Prepatation and Inputs -----
    # ---------------------------------

    # --- Hamiltonian Parameters for Test ---
    lmbda = 4*np.pi; mQ = 1.; r = 1.; e_max=7; lmaxeff = effective_lmax(e_max,e_max,mQ,r)
    # Compute prefactor
    cphi4 = lmbda*(1/(8*math.pi))*(1/r)


    # ---------------------------------
    # -- Basis and Matrix Generation --
    # ---------------------------------

    # --- Generate the Basis ---
    dict_basis, hashable_basis, iterSet = basis_even(*gen_basis(lmaxeff,e_max,mQ,r)) #Select Even Basis States Only.
    # Generate useful lists of energies.
    gen_omega_list(lmaxeff, mQ, r)

    # --- Run the transition matrix element calculation in phi^4 ---
    H_eff_coo=transition_matrix_phi4(mQ, r, cphi4, lmaxeff, dict_basis, hashable_basis, iterSet)

    # ---------------------------------
    # ------- Plot Preparation --------
    # ---------------------------------

    full_basis=dict_basis

    # 1. Compute properties.
    occupation_numbers = np.array([get_total_occupation_number(state) for state in full_basis]) # State Occupation Number for each in basis.
    full_state_energies = np.array([get_state_energy(state, mQ, r) for state in full_basis]) ## State Energies for each in basis.


    # 2. Sort the basis using lexsort, primarily by occupation number, secondly by energy.
    double_sorted_indices = np.lexsort((full_state_energies, occupation_numbers)) 
    sorted_full_basis = [full_basis[i] for i in double_sorted_indices]
    H_sorted = H_eff_coo.toarray()[double_sorted_indices, :][:, double_sorted_indices]


    # 3. Identify unique occupation values and their starting indices
    # (Using the sorted array guarantees these indices mark the boundaries)
    unique_occ, boundaries_vec = np.unique(occupation_numbers[double_sorted_indices], return_index=True)

    # 4. Map the boundaries to a dictionary
    boundaries = dict(zip(unique_occ, boundaries_vec))


    # --- --- Plot --- ---

    # --- State labels ---
    verbose_labels = []
    for state in sorted_full_basis:
    #for state in full_basis:
        verbose_labels.append("{"+", ".join(f"{key}: {value}" for key, value in sorted(state.items()) if value !=0)+"}") # Such that they 'look' like a dictionary of states on the plot.

    # --- Boundaries between occupation numbers for the plot ---
    state_vecs = np.array([convert_state_to_list(state, lmaxeff) for state in sorted_full_basis])
    boundaries = np.where(np.diff(np.sort(np.sum(state_vecs, axis=1))))[0] # List of places where state occupation number changes.
    boundaries[0] = 0 # Fixing the zero entry.

    # --- Plotting the Heatmap Matrix ---
    plt.figure(figsize=(12, 12))

    zero_mask = (H_sorted == 0.0) # So that we don't have an overcrowded plot.
    ax = sns.heatmap(
        H_sorted, 
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

    plt.tight_layout()
    plt.show()
