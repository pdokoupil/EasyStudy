import numpy as np

NEG_INF = int(-10e6)
P_MUTATION = 1.0
P_CROSSOVER = 0.3
POP_SIZE = 10

def mask_scores(scores, seen_items_mask):
    # Ensure seen items get lowest score of 0
    # Just multiplying by zero does not work when scores are not normalized to be always positive
    # because masked-out items will not have smallest score (some valid, non-masked ones can be negative)
    # scores = scores * seen_items_mask[user_idx]
    # So instead we do scores = scores * seen_items_mask[user_idx] + NEG_INF * (1 - seen_items_mask[user_idx])
    min_score = scores.min()
    # Unlike in predict_with_score, here we do not mandate NEG_INF to be strictly smaller
    # because rel_scores may already contain some NEG_INF that was set by predict_with_score
    # called previously -> so we allow <=.
    assert NEG_INF <= min_score, f"min_score ({min_score}) is not smaller than NEG_INF ({NEG_INF})"
    scores = scores * seen_items_mask + NEG_INF * (1 - seen_items_mask)
    return scores


##### Helpers needed for evolutionary algorithms #####
from typing import List

from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file, \
    print_variables_to_file

from jmetal.operator.selection import RouletteWheelSelection, BinaryTournamentSelection
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.operator.crossover import IntegerSBXCrossover

from jmetal.algorithm.multiobjective import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations, StoppingByTime

from jmetal.core.problem import IntegerProblem
from jmetal.core.solution import IntegerSolution

class RecommendationProblemExact(IntegerProblem):
    def __init__(self, k, objs, user_idx, relevance_scores, user_vector, relevance_top_k, filter_out_items, target_weights):
        super(RecommendationProblemExact, self).__init__()
        
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.number_of_variables = k

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["dist"]

        # Total number of items (before filtering seen items)
        assert relevance_scores.ndim == 1
        n_all_items = relevance_scores.size
        self.items = np.setdiff1d(np.arange(n_all_items), filter_out_items)
        # Number of items after filtering!
        self.n_items = self.items.size

        assert self.n_items <= n_all_items, f"{self.n_items} <= {n_all_items}"
        self.item_indices = np.arange(self.n_items)

        self.lower_bound = [0] * self.number_of_variables
        # Upper bound equals new number of items (i.e. after filtering)
        self.upper_bound = [self.n_items - 1] * self.number_of_variables

        self.user_idx = user_idx
        self.objs = objs
        self.relevance_top_k = relevance_top_k
    
        self.user_vector = user_vector
        assert user_vector.ndim == 1 and user_vector.size == n_all_items

        self.target_weights = target_weights

    # Evaluate solution w.r.t. the objectives
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        top_k_list = solution.variables
        
        # During evaluation we need to map items, note that item 0 corresponds to actual item being self.items[0] (due to filtering)
        # We need to perform evaluation on original items instead of just their indices
        top_k_mapped = self.items[top_k_list]
        obj_vals = np.zeros(shape=(len(self.objs),), dtype=np.float32)
        for i, obj in enumerate(self.objs):
            obj_vals[i] = obj(top_k_mapped)
        
        # Our objective is distance between actual and target weights
        solution.objectives = [((obj_vals - self.target_weights) ** 2).sum()]

        assert solution is not None
        return solution

    # We should constraint the new solution to never contain duplicates
    # and also update mutation and crossover accordingly (to not introduce them later on in the optimization process)
    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives, self.number_of_constraints
        )
 
        #new_solution.variables = np.random.choice(self.items, size=self.number_of_variables, replace=False).tolist()
        #### NEW ####
        # Instead of generating new solution from random variables, we start with relevance only recommendation
        new_solution.variables = self.relevance_top_k.copy()
        rnd_idx = np.random.randint(low=0, high=self.relevance_top_k.size)
        rnd_item = np.random.choice(self.item_indices)
        new_solution.variables[rnd_idx] = rnd_item
        return repair_duplicates(self.item_indices, new_solution)
    
    def number_of_variables(self):
        return self.number_of_variables
    
    def number_of_objectives(self):
        return self.number_of_objectives
    
    def number_of_constraints(self):
        return self.number_of_constraints
    
    def name(self) -> str:
        return self.__name__

# Abstraction of Recommendation problem (as integer problem from jmetalpy)
class RecommendationProblemMax(IntegerProblem):

    def __init__(self, k, objs, user_idx, relevance_scores, user_vector, relevance_top_k, filter_out_items):
        super(RecommendationProblemMax, self).__init__()

        self.number_of_objectives = len(objs)
        self.number_of_constraints = 0
        self.number_of_variables = k
        
        # Total number of items (before filtering seen items)
        assert relevance_scores.ndim == 1
        n_all_items = relevance_scores.size

        self.items = np.setdiff1d(np.arange(n_all_items), filter_out_items)
        # Number of items after filtering!
        self.n_items = self.items.size
        self.relevance_scores = relevance_scores

        assert self.n_items <= n_all_items, f"{self.n_items} <= {n_all_items}"
        self.item_indices = np.arange(self.n_items)
        
        self.user_idx = user_idx
        self.objs = objs
        self.relevance_top_k = relevance_top_k
    
        self.user_vector = user_vector
        assert user_vector.ndim == 1 and user_vector.size == n_all_items
        # By default we maximize all the objectives in recommendation
        self.obj_directions = [self.MAXIMIZE] * len(objs)
        self.obj_labels = [o.name() for o in objs]
        
        self.lower_bound = [0] * self.number_of_variables
        # Upper bound equals new number of items (i.e. after filtering)
        self.upper_bound = [self.n_items - 1] * self.number_of_variables

    # Evaluate solution w.r.t. the objectives
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        top_k_list = solution.variables
        
        # During evaluation we need to map items, note that item 0 corresponds to actual item being self.items[0] (due to filtering)
        # We need to perform evaluation on original items instead of just their indices
        top_k_mapped = self.items[top_k_list]
        for i, obj in enumerate(self.objs):
            solution.objectives[i] = obj(top_k_mapped)
        
        assert solution is not None
        return solution

    # We should constraint the new solution to never contain duplicates
    # and also update mutation and crossover accordingly (to not introduce them later on in the optimization process)
    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives, self.number_of_constraints
        )
 
        #new_solution.variables = np.random.choice(self.items, size=self.number_of_variables, replace=False).tolist()
        #### NEW ####
        # Instead of generating new solution from random variables, we start with relevance only recommendation
        new_solution.variables = self.relevance_top_k.copy()
        rnd_idx = np.random.randint(low=0, high=self.relevance_top_k.size)
        rnd_item = np.random.choice(self.item_indices)
        new_solution.variables[rnd_idx] = rnd_item
        return repair_duplicates(self.item_indices, new_solution)
        
        
        #return new_solution
    #def create_solution(self) -> IntegerSolution:
    #    new_solution = IntegerSolution(lower_bounds, upper_bounds, number_of_constraints=self.number_of_constraints,
    #                              number_of_objectives=self.number_of_objectives)

    def number_of_variables(self):
        return self.number_of_variables
    
    def number_of_objectives(self):
        return self.number_of_objectives
    
    def number_of_constraints(self):
        return self.number_of_constraints
    
    def name(self) -> str:
        return self.__name__

# Function for repairing integer solution by getting rid of duplicated entries
def repair_duplicates(all_items, res: IntegerSolution):
    # Those are items not in the recommendation yet
    candidate_items = np.setdiff1d(all_items, res.variables)
    for i in range(len(res.variables)):
        if res.variables[i] in res.variables[i+1:]:
            rnd_candidate_index = np.random.randint(low=0, high=candidate_items.size)
            res.variables[i] = candidate_items[rnd_candidate_index]
            # Delete the candidate
            np.delete(candidate_items, rnd_candidate_index)
    return res

# Same as IntegerPolynomialMutation, but does "repair" the solution
# by removing duplicates
class FixingMutation(IntegerPolynomialMutation):
    def __init__(self, items, relevance_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = items
    
    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        res = super().execute(solution)
        return repair_duplicates(self.items, res)
    
# Same as IntegerSBXCrossover, but does "repair" the solution
# by removing duplicates
class FixingCrossover(IntegerSBXCrossover):
    def __init__(self, items, relevance_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = items

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        res = super().execute(parents)
        assert len(res) == 2
        
        res[0] = repair_duplicates(self.items, res[0])
        res[1] = repair_duplicates(self.items, res[1])
        return res

##### End of evolutionary helpers ######

class evolutionary_max:
    def __init__(self, weights, relevance_estimates, user_vector, objectives, relevance_top_k, filter_out_items, k=10, time_limit_seconds=4):
        self.weights = weights
        self.p_mutation = P_MUTATION
        self.p_crossover = P_CROSSOVER
        self.problem = RecommendationProblemMax(
            relevance_scores=relevance_estimates,
            k=k,
            objs=objectives,
            user_idx=0, # We expect a single user
            user_vector=user_vector,
            relevance_top_k=relevance_top_k,
            filter_out_items=filter_out_items
        )

        self.algorithm = NSGAII(
            problem=self.problem,
            population_size=10,
            offspring_population_size=10,
            mutation=FixingMutation(self.problem.item_indices, relevance_estimates, probability=self.p_mutation, distribution_index=20),
            crossover=FixingCrossover(self.problem.item_indices, relevance_estimates, probability=self.p_crossover, distribution_index=20),
            #mutation=IntegerPolynomialMutation(probability=p_mutation),
            #crossover=IntegerSBXCrossover(probability=p_crossover),
            #termination_criterion=StoppingByEvaluations(max_evaluations=25000),
            termination_criterion=StoppingByTime(max_seconds=time_limit_seconds),
            selection=RouletteWheelSelection()
        )
    
    def select_single_solution(self, front):
        min_dist = np.inf
        min_dist_idx = None

        for idx, solution in enumerate(front):
            dist = ((solution.objectives - self.weights) ** 2).sum()
            if min_dist_idx is None or dist < min_dist:
                min_dist_idx = idx
                min_dist = dist
        
        return front[min_dist_idx]

    def __call__(self):
        self.algorithm.run()
        front = get_non_dominated_solutions(self.algorithm.get_result())
        best_solution = self.select_single_solution(front)
        # Do the item index mapping
        return self.problem.items[best_solution.variables]

class evolutionary_exact:
    def __init__(self, weights, relevance_estimates, user_vector, objectives, relevance_top_k, filter_out_items, k=10, time_limit_seconds=4):
        self.weights = weights
        self.p_mutation = P_MUTATION
        self.p_crossover = P_CROSSOVER
        self.problem = RecommendationProblemExact(
            relevance_scores=relevance_estimates,
            k=k,
            objs=objectives,
            user_idx=0, # We expect a single user
            user_vector=user_vector,
            relevance_top_k=relevance_top_k,
            filter_out_items=filter_out_items,
            target_weights=weights
        )

        self.algorithm = NSGAII(
            problem=self.problem,
            population_size=10,
            offspring_population_size=10,
            mutation=FixingMutation(self.problem.item_indices, relevance_estimates, probability=self.p_mutation, distribution_index=20),
            crossover=FixingCrossover(self.problem.item_indices, relevance_estimates, probability=self.p_crossover, distribution_index=20),
            #mutation=IntegerPolynomialMutation(probability=p_mutation),
            #crossover=IntegerSBXCrossover(probability=p_crossover),
            #termination_criterion=StoppingByEvaluations(max_evaluations=25000),
            termination_criterion=StoppingByTime(max_seconds=time_limit_seconds),
            selection=RouletteWheelSelection()
        )

    def __call__(self):
        self.algorithm.run()
        front = get_non_dominated_solutions(self.algorithm.get_result())
        assert len(front) == 1, f'We expect single solution when doing single objective optimization, got: {len(front)}'
        best_solution = front[0]
        # Do the item index mapping
        return self.problem.items[best_solution.variables]

class item_wise_exact:
    def __init__(self, weights):
        self.weights = weights
        self.n_objectives = weights.size

    def __call__(self, mgains, seen_items_mask):
        # Greedy algorithm that just search items whose supports are
        _, n_items = mgains.shape
        assert self.n_objectives == mgains.shape[0], f"{self.n_objectives} != {mgains.shape}"
        # Convert distances to scores (the lower the distance, the higher the score)
        scores = -1 * ((mgains - self.weights[:, np.newaxis]) ** 2).sum(axis=0)
        assert scores.shape == (n_items, ), f"{scores.shape} != {(n_items, )}"
        print(f"Max score without masking = {scores.max()}, mgains: {mgains[:, scores.argmax()]}")
        print(mgains)
        scores = mask_scores(scores, seen_items_mask)
        print(f"Max score = {scores.max()}, mgains: {mgains[:, scores.argmax()]}")
        return scores.argmax()
    
class item_wise_max:
    def __init__(self, weights):
        self.weights = weights
        self.n_objectives = weights.size

    def __call__(self, mgains, seen_items_mask):
        # Greedy algorithm that just search items whose supports are
        _, n_items = mgains.shape
        assert self.n_objectives == mgains.shape[0], f"{self.n_objectives} != {mgains.shape}"
        # Convert distances to scores (the lower the distance, the higher the score)
        scores = (mgains * self.weights[:, np.newaxis]).sum(axis=0)
        assert scores.shape == (n_items, ), f"{scores.shape} != {(n_items, )}"
        print(f"Max score without masking = {scores.max()}, mgains: {mgains[:, scores.argmax()]}")
        print(mgains)
        scores = mask_scores(scores, seen_items_mask)
        print(f"Max score = {scores.max()}, mgains: {mgains[:, scores.argmax()]}")
        return scores.argmax()

# TODO modify
class greedy_exact:
    def __init__(self, weights):
        self.weights = weights
        self.n_objectives = weights.size
        self.TOT = 0.0
        self.gm = np.zeros(shape=(self.n_objectives, ), dtype=np.float32)

    # Returns single item recommended at a given time
    def __call__(self, mgains, seen_items_mask):
        _, n_items = mgains.shape
        assert np.all(mgains >= 0.0), "We expect all mgains to be normalized to be greater than 0"
        tots = np.zeros(shape=(n_items, ), dtype=np.float32)
        remainder = np.zeros_like(self.gm)
        scores = np.zeros_like(tots)
        for item_idx in np.arange(n_items):
            tots[item_idx] = max(self.TOT, self.TOT + mgains[:, item_idx].sum())
            remainder = tots[item_idx] * self.weights - self.gm
            assert remainder.shape == (self.n_objectives,)
            # We want to push remainder to 0 by maximizing score (so we minimize its negative)
            scores[item_idx] = -((remainder - mgains[:, item_idx]) ** 2).sum()

        scores = mask_scores(scores, seen_items_mask)
        i_best = scores.argmax()

        self.gm = self.gm + mgains[:, i_best]
        self.TOT = np.clip(self.gm, 0.0, None).sum()

        return i_best

# Original rlprop algorithm
class greedy_max:
    def __init__(self, weights):
        self.weights = weights
        self.n_objectives = weights.size
        self.TOT = 0.0
        self.gm = np.zeros(shape=(self.n_objectives, ), dtype=np.float32)

    # Returns single item recommended at a given time
    def __call__(self, mgains, seen_items_mask):
        _, n_items = mgains.shape
        assert np.all(mgains >= 0.0), "We expect all mgains to be normalized to be greater than 0"
        tots = np.zeros(shape=(n_items, ), dtype=np.float32)
        remainder = np.zeros_like(self.gm)
        scores = np.zeros_like(tots)
        for item_idx in np.arange(n_items):
            tots[item_idx] = max(self.TOT, self.TOT + mgains[:, item_idx].sum())
            remainder = tots[item_idx] * self.weights - self.gm
            assert remainder.shape == (self.n_objectives,)
            
            positive_gain_mask = mgains >= 0.0
            negative_gain_mask = mgains < 0.0
            scores[positive_gain_mask] = np.maximum(0, np.minimum(mgains[positive_gain_mask], remainder[positive_gain_mask]))
            scores[negative_gain_mask] = np.minimum(0, mgains[negative_gain_mask] - remainder[negative_gain_mask])
            
        scores = mask_scores(scores, seen_items_mask)
        i_best = scores.argmax()

        self.gm = self.gm + mgains[:, i_best]
        self.TOT = np.clip(self.gm, 0.0, None).sum()

        return i_best