"""
Define default VMB module
"""

import numpy as np
import starsim as ss
import sciris as sc
import datetime as dt
from dataclasses import dataclass

ss_int_ = ss.dtypes.int

__all__ = ['VMB']

@dataclass
class Par:
    data: dict

    def to_dict(self):
        return self.data

    def __repr__(self):
        return str(self.data)

#### Module code ####
class VMB(ss.Module):

    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()

        # Parameters
        self.define_pars(
            unit='day',
            menstrual_hygiene = sc.objdict(
                products = Par(data={'cup': 0.05, 'single_use': 0.4, 'reusable_homemade': 0.35, 'reusable_manufactured': 0.2}),  # could be adjusted for urbanicity/SES
                # Weighted likelihoods: if you use a reusable homemade product higher likelihood you have poor WASH access and are prone to overuse
                WASH = sc.objdict(
                    cup = Par(data={'good': 0.6, 'medium': 0.3, 'poor': 0.1}),
                    single_use = Par(data={'good': 0.7, 'medium': 0.2, 'poor': 0.1}),
                    reusable_homemade = Par(data={'good': 0.1, 'medium': 0.3, 'poor': 0.6}),  # Higher likelihood of poor WASH access
                    reusable_manufactured = Par(data={'good': 0.4, 'medium': 0.3, 'poor': 0.3})
                ),
                 # normal = appropriate change rate, over = uLile to change at appropriate rate
                normal_use = sc.objdict(
                    cup = ss.bernoulli(p=0.6),
                    single_use = ss.bernoulli(p=0.7),
                    reusable_homemade = ss.bernoulli(p=0.3),  # Higher likelihood of overuse
                    reusable_manufactured = ss.bernoulli(p=0.6)
                ),
                # Borzutzky & Jaffray, 2019 2. CDC – Heavy Menstrual Bleeding 3. O'Brien et al., 2019; Vo et al., 2013 4. Sinharoy et al. 2024
                flow = Par(data={'light': 0.25, 'medium': 0.25, 'heavy': 0.5})
            ),
            sex_practices = sc.objdict(
                condom_use = ss.bernoulli(p=0.39),
                circumcision_status = ss.bernoulli(p=0.49),
            ),
            birth_control = Par(data={'iud': 0.1, 'DEPO': 0.2, 'pill': 0.15, 'condom': 0.25, 'none': 0.3}),  # rough estimates
            antibiotics =ss.bernoulli(p= 0.1),  # Probability of taking antibiotics
            bacteria_types = ['nAB', 'Li', 'oLB']
            
        )
        self.update_pars(pars=pars, **kwargs)

        self.define_states(

            ss.FloatArr('oLB', 0), # absolute abundance of Lactobacillus crispatus (oLB)
            ss.FloatArr('Li', 0), # absolute abundance of Lactobacillus iners (Li)
            ss.FloatArr('nAB', 0), # absolute abundance of non-optimal anaerobes (nAB)

            # VMB states
            ss.State('CST_I'),                   
            ss.State('CST_III'),                 
            ss.State('CST_IV'),                    

        )

        return

    @staticmethod
    def generate_random_composition(bacteria_types, initial_biomass=1000, min_abundance=100):
        dominance_likelihoods = {
            'oLB': 0.10,
            'Li': 0.35,
            'nAB': 0.55,
        }

        # Ensure initial biomass is sufficient to meet minimum requirement for all bacteria
        total_min_abundance = min_abundance * len(bacteria_types)
        if initial_biomass < total_min_abundance:
            raise ValueError("Initial biomass is too low to meet the minimum abundance requirement for all bacteria.")

        # Choose a dominant bacterium based on likelihoods
        dominant_bacterium = np.random.choice(bacteria_types, p=[dominance_likelihoods[bt] for bt in bacteria_types])

        # Randomly assign a dominant proportion of remaining biomass
        dominant_proportion = np.random.uniform(0.6, 0.8)
        remaining_proportion = 1.0 - dominant_proportion

        # Allocate minimum abundance to each bacterium
        initial_allocations = {bt: min_abundance for bt in bacteria_types}
        remaining_biomass = initial_biomass - sum(initial_allocations.values())

        # Distribute remaining proportion among other bacteria
        other_bacteria = [bt for bt in bacteria_types if bt != dominant_bacterium]
        other_proportions = np.random.dirichlet(np.ones(len(other_bacteria))) * remaining_proportion

        # Adjust proportions to include minimum abundance
        proportions = {bt: initial_allocations[bt] / initial_biomass for bt in bacteria_types}
        proportions.update({bt: proportions[bt] + (prop * remaining_biomass / initial_biomass) for bt, prop in zip(other_bacteria, other_proportions)})
        proportions[dominant_bacterium] = (initial_allocations[dominant_bacterium] + (dominant_proportion * remaining_biomass)) / initial_biomass

        # Calculate absolute abundance
        absolute_abundance = {bt: proportions[bt] * initial_biomass for bt in bacteria_types}

        return proportions, absolute_abundance
    
    def assign_CST(self):
        dominant_bacterium = max(self.proportions, key=self.proportions.get)
        self.CST = {
            'oLB': 'CST I',
            'Li': 'CST III',
            'nAB': 'CST IV',
        }.get(dominant_bacterium, 'Unknown CST')
        return self.CST

    
    def init_results(self):
        """ Initialize results """
        super().init_results()
        return


    def init_post(self):
        """
        Set initial values for states. This could involve passing in a full set of initial conditions,
        or using init_prev, or other. Note that this is different to initialization of the State objects
        i.e., creating their dynamic array, linking them to a People instance. That should have already
        taken place by the time this method is called.
        """
        super().init_post()

        # Set initial VMB for population
        self.proportions, self.absolute_abundance = self.generate_random_composition(self.pars['bacteria_types'])
        self.CST = self.assign_CST() 

        return

    def update_results(self):
        """ Update results """
        super().update_results()
        return

    def start_step(self):
        """
        Updates that take place at the beginning of a time step, prior to interventions and transmission
        To understand what happens during a time step, look at the collect_funcs() function in loop.py of starsim.
        """
        super().start_step()
        return

    def step_state(self):
        # State transitions (before interventions and transmission)
        super().step_state()
        return

    def step(self):
        # State transitions (after interventions and transmission)
        super().step()

        return


