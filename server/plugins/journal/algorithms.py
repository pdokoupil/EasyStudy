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

# For exact, we treat recommendation problem as single-objective optimization, where the single objective
# corresponds to distance between weights and objectives evaluated on top of each solution
class RecommendationProblemExact(IntegerProblem):
    def __init__(self, k, objs, user_idx, relevance_scores, user_vector, relevance_top_k, filter_out_items, target_weights, mgain_cdf):
        super(RecommendationProblemExact, self).__init__()
        
        # Single objective that corresponds to distance between weights and actual objectives
        self.number_of_objectives = 1
        # No constraints
        self.number_of_constraints = 0
        # Number of variables corresponds to length of recommendation list
        self.number_of_variables = k

        # We minimize the objective (distance)
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["dist"]

        assert relevance_scores.ndim == 1
        
        # Total number of items (before filtering seen items)
        n_all_items = relevance_scores.size
        # Filter out the items
        self.items = np.setdiff1d(np.arange(n_all_items), filter_out_items)
        # Number of items after filtering! This is the number we choose from in individuals
        self.n_items = self.items.size

        # Maping from items to item_indices used internally by this problem
        self.inverse_mapping = { item: item_idx for item_idx, item in enumerate(self.items) }

        assert self.n_items <= n_all_items, f"{self.n_items} <= {n_all_items}"
        # We need consecutive numbers for the items (we are sampling from [lower_bound, upper_bound))
        self.item_indices = np.arange(self.n_items)

        # Lower bound is 0
        self.lower_bound = [0] * self.number_of_variables
        # Upper bound equals new number of items (i.e. after filtering)
        self.upper_bound = [self.n_items - 1] * self.number_of_variables

        # Index of the user for which we generate the recommendation
        self.user_idx = user_idx

        # The actual objectives that WE want to optimize (not the objective that is optimized by MO-EA algorithm as that is only proxy to our objectives)
        self.objs = objs
        # Relevance based recommendation
        self.relevance_top_k = relevance_top_k
        # User vector (binary feedback from the user)
        self.user_vector = user_vector
        assert user_vector.ndim == 1 and user_vector.size == n_all_items
        # Target weights we are interested in achieving in our recommendation
        self.target_weights = target_weights

        # CDF normalization for marginal gains
        self.mgain_cdf = mgain_cdf

        # Calculate marginal gain of each of the k items w.r.t. each objective
        # we allocate the array here so we can reuse it in evaluate
        self.mgains = np.zeros(shape=(len(self.objs), k))

    # Evaluate solution w.r.t. the objectives
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        # The variables of solution encode the recommendation
        top_k_list = solution.variables

        # The problem here is that we need to normalize objectives to calculate distance w.r.t. weights
        # properly. However, we do not have right normalization available that would work for whole lists.
        # For that reason, we resort to use approximation by calculating normalized marginal gains
        # for individual items in the recommendation and representing whole recommendation as an average distance
        # over the individual items.
        
        # During evaluation we need to map items, note that item 0 corresponds to actual item being self.items[0] (due to filtering)
        # We need to perform evaluation on original items instead of just their indices
        # Evaluate all the objectives on the given recommendation list
        top_k_mapped = self.items[top_k_list]

        for item_idx, item_mapped in enumerate(top_k_mapped):
            # Get marginal gains
            for obj_idx, obj in enumerate(self.objs):
                self.mgains[obj_idx, item_idx] = obj(top_k_mapped[:item_idx+1]) - obj(top_k_mapped[:item_idx])

        # Normalize the marginal gains
        # We need to do double transpose, first we want mgains (n_objs, n_items) to be (n_items, n_objs) and then we transform it back
        mgains = self.mgain_cdf.transform(self.mgains.T).T
        # Get distance of the items
        item_distances = ((mgains - self.target_weights[:, np.newaxis]) ** 2).sum(axis=0)
        # MO-EA's objective is mean distance over the items from the recommendation
        solution.objectives = [item_distances.mean()]

        assert solution is not None
        return solution

    # We should constraint the new solution to never contain duplicates
    # and also update mutation and crossover accordingly (to not introduce them later on in the optimization process)
    def create_solution(self) -> IntegerSolution:
        # Create new solution (integer solution with given bounds)
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives, self.number_of_constraints
        )
 
        # Instead of generating new solution from random variables, we start with relevance only recommendation
        new_solution.variables = np.array([self.inverse_mapping[item] for item in self.relevance_top_k])
        # And replace position at random index with a random item
        rnd_idx = np.random.randint(low=0, high=self.relevance_top_k.size)
        rnd_item = np.random.choice(self.item_indices)
        new_solution.variables[rnd_idx] = rnd_item
        # Finally we repair duplicates (no duplicates are allowed)
        return repair_duplicates(self.n_items, new_solution)
    
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

    def __init__(self, k, framework_objectives, user_idx, relevance_scores, user_vector, relevance_top_k, filter_out_items):
        super(RecommendationProblemMax, self).__init__()

        # Number of MO-EA's objective is same as our objectives
        self.number_of_objectives = len(framework_objectives)
        # We do not have any constraints
        self.number_of_constraints = 0
        # Number of variables corresponds to the length of the recommendation list
        self.number_of_variables = k

        assert relevance_scores.ndim == 1

        # Total number of items (before filtering seen items)
        n_all_items = relevance_scores.size
        # Filter out the items
        self.items = np.setdiff1d(np.arange(n_all_items), filter_out_items)
        # Number of items after filtering!
        self.n_items = self.items.size
        # Relevances (normalized so they are >= 0) predicted by relevance baseline for each of the items
        self.relevance_scores = relevance_scores

        # Maping from items to item_indices used internally by this problem
        self.inverse_mapping = { item: item_idx for item_idx, item in enumerate(self.items) }

        assert self.n_items <= n_all_items, f"{self.n_items} <= {n_all_items}"
        # We need consecutive numbers for the items (we are sampling from [lower_bound, upper_bound))
        self.item_indices = np.arange(self.n_items)
        
        # Index of the user for which we generate the recommendation
        self.user_idx = user_idx
        # Save the objectives
        self.framework_objectives = framework_objectives
        # Relevance-only recommendation list
        self.relevance_top_k = relevance_top_k
    
        # User vector with implicit feedback
        self.user_vector = user_vector
        assert user_vector.ndim == 1 and user_vector.size == n_all_items

        # By default we maximize all the objectives in recommendation
        # self.obj_directions = [self.MAXIMIZE] * len(framework_objectives)
        self.obj_labels = [o.name() for o in framework_objectives]
        
        # Lower bound is 0
        self.lower_bound = [0] * self.number_of_variables
        # Upper bound equals new number of items (i.e. after filtering)
        self.upper_bound = [self.n_items - 1] * self.number_of_variables

    # Evaluate solution w.r.t. the objectives
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        top_k_list = solution.variables
        # During evaluation we need to map items, note that item 0 corresponds to actual item being self.items[0] (due to filtering)
        # We need to perform evaluation on original items instead of just their indices
        # We evaluate framework objetives here
        top_k_mapped = self.items[top_k_list]
        for i, obj in enumerate(self.framework_objectives):
            solution.objectives[i] = 1 / (1 + obj(top_k_mapped)) # We need to maximize
        assert solution is not None
        return solution

    # We should constraint the new solution to never contain duplicates
    # and also update mutation and crossover accordingly (to not introduce them later on in the optimization process)
    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives, self.number_of_constraints
        )
 
        # Instead of generating new solution from random variables, we start with relevance only recommendation
        new_solution.variables = np.array([self.inverse_mapping[item] for item in self.relevance_top_k])
        # And we replace item at random position with a random item
        rnd_idx = np.random.randint(low=0, high=self.relevance_top_k.size)
        rnd_item = np.random.choice(self.item_indices)
        new_solution.variables[rnd_idx] = rnd_item
        # Finaly we get rid of all duplicates as these are prohibited
        return repair_duplicates(self.n_items, new_solution)

    def number_of_variables(self):
        return self.number_of_variables
    
    def number_of_objectives(self):
        return self.number_of_objectives
    
    def number_of_constraints(self):
        return self.number_of_constraints
    
    def name(self) -> str:
        return self.__name__

# Function for repairing integer solution by getting rid of duplicated entries
# Whenever we see an item that is present multiple times, we start getting rid of its occurrences (while we have > 1 of them)
# by replacing these items by randomly sampled items
def repair_duplicates(n_items, res: IntegerSolution):
    unique_values = set()
    for i in range(len(res.variables)):
        while res.variables[i] in unique_values:
            res.variables[i] = np.random.randint(0, n_items)
        unique_values.add(res.variables[i])
    return res

# Same as IntegerPolynomialMutation, but does "repair" the solution
# by removing duplicates
class FixingMutation(IntegerPolynomialMutation):
    def __init__(self, n_items, relevance_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_items = n_items
    
    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        res = super().execute(solution)
        return repair_duplicates(self.n_items, res)
    
# Same as IntegerSBXCrossover, but does "repair" the solution
# by removing duplicates
class FixingCrossover(IntegerSBXCrossover):
    def __init__(self, n_items, relevance_scores, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_items = n_items

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        res = super().execute(parents)
        assert len(res) == 2
        
        res[0] = repair_duplicates(self.n_items, res[0])
        res[1] = repair_duplicates(self.n_items, res[1])
        return res

##### End of evolutionary helpers ######

# Actual evolutionary algorithm that wraps the underlying helpers passed to MO-EA jmetalpy framework
class evolutionary_max:
    def __init__(self, mgain_cdf, weights, relevance_estimates, user_vector, relevance_top_k, filter_out_items, our_objectives, framework_objectives, k=10, time_limit_seconds=4):
        # our and framework objectives have to have the same length
        assert len(our_objectives) == len(framework_objectives)
        
        self.weights = weights
        self.p_mutation = P_MUTATION
        self.p_crossover = P_CROSSOVER
        self.problem = RecommendationProblemMax(
            relevance_scores=relevance_estimates,
            k=k,
            framework_objectives=framework_objectives,
            user_idx=0, # We expect a single user
            user_vector=user_vector,
            relevance_top_k=relevance_top_k,
            filter_out_items=filter_out_items
        )

        self.algorithm = NSGAII(
            problem=self.problem,
            population_size=10,
            offspring_population_size=10,
            mutation=FixingMutation(self.problem.n_items, relevance_estimates, probability=self.p_mutation, distribution_index=20),
            crossover=FixingCrossover(self.problem.n_items, relevance_estimates, probability=self.p_crossover, distribution_index=20),
            #mutation=IntegerPolynomialMutation(probability=p_mutation),
            #crossover=IntegerSBXCrossover(probability=p_crossover),
            #termination_criterion=StoppingByEvaluations(max_evaluations=25000),
            termination_criterion=StoppingByTime(max_seconds=time_limit_seconds),
            selection=RouletteWheelSelection()
        )

        self.mgain_cdf = mgain_cdf
        self.our_objectives = our_objectives
        self.k = k
    
    def select_single_solution(self, front):
        min_dist = np.inf
        min_dist_idx = None

        item_distances = np.zeros(shape=(self.k), dtype=np.float32)
        # Calculate marginal gain of each of the k items w.r.t. each objective
        mgains = np.zeros(shape=(len(self.our_objectives), self.k))

        for idx, solution in enumerate(front):
            assert len(solution.variables) == self.k
            # The problem here is that we need to normalize objectives to calculate distance w.r.t. weights
            # properly. However, we do not have right normalization available that would work for whole lists.
            # For that reason, we resort to use approximation by calculating normalized marginal gains
            # for individual items in the recommendation and representing whole recommendation as an average distance
            # over the individual items.
            variables_mapped = self.problem.items[solution.variables]
            for item_idx, item_mapped in enumerate(variables_mapped):
                # Get marginal gains
                for obj_idx, obj in enumerate(self.our_objectives):
                    mgains[obj_idx, item_idx] = obj(variables_mapped[:item_idx+1]) - obj(variables_mapped[:item_idx])

            # Normalize the marginal gains
            # We need to do double transpose, first we want mgains (n_objs, n_items) to be (n_items, n_objs) and then we transform it back
            mgains = self.mgain_cdf.transform(mgains.T).T
            # Get distance of the items
            item_distances = ((mgains - self.weights[:, np.newaxis]) ** 2).sum(axis=0)

            dist = item_distances.mean()
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
    def __init__(self, mgain_cdf, weights, relevance_estimates, user_vector, objectives, relevance_top_k, filter_out_items, k=10, time_limit_seconds=4):
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
            target_weights=weights,
            mgain_cdf=mgain_cdf
        )

        self.algorithm = NSGAII(
            problem=self.problem,
            population_size=10,
            offspring_population_size=10,
            mutation=FixingMutation(self.problem.n_items, relevance_estimates, probability=self.p_mutation, distribution_index=20),
            crossover=FixingCrossover(self.problem.n_items, relevance_estimates, probability=self.p_crossover, distribution_index=20),
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

# Basic implementation that calculates scores as (1 - alpha) * mgain_rel + alpha * mgain_div
# that is used in diversification experiments
class unit_normalized_diversification:
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, mgains, seen_items_mask):
        # We ignore seen items mask here
        assert mgains.ndim == 2 and mgains.shape[0] == 2, f"shape={mgains.shape}"
        scores = (1.0 - self.alpha) * mgains[0] + self.alpha * mgains[1]
        assert scores.ndim == 1 and scores.shape[0] == mgains.shape[1], f"shape={scores.shape}"
        scores = mask_scores(scores, seen_items_mask)
        return scores.argmax()

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

class greedy_exact:
    def __init__(self, weights):
        self.weights = weights
        self.n_objectives = weights.size
        self.TOT = np.zeros(shape=(self.n_objectives, ), dtype=np.float32)

    # Returns single item recommended at a given time
    def __call__(self, mgains, seen_items_mask):
        assert np.all(mgains >= 0.0), "We expect all mgains to be normalized to be greater than 0"

        scores = -(self.TOT[:, np.newaxis] + ((mgains - self.weights[:, np.newaxis]) ** 2)).sum(axis=0)
        assert scores.shape == (mgains.shape[1], )
        scores = mask_scores(scores, seen_items_mask)

        i_best = scores.argmax()

        self.TOT = (self.TOT[:, np.newaxis] + self.weights[:, np.newaxis] - mgains)[:, i_best]
        assert self.TOT.shape == (self.n_objectives, ), f"Shape={self.TOT.shape}"
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
        # Added second dimension to enable broadcasting
        remainder = np.zeros_like(self.gm)[:, np.newaxis]
        gain_items = np.zeros_like(mgains)
        for item_idx in np.arange(n_items):
            tots[item_idx] = max(self.TOT, self.TOT + mgains[:, item_idx].sum())
            remainder[:, 0] = tots[item_idx] * self.weights - self.gm
            assert remainder.shape == (self.n_objectives, 1)
            
            positive_gain_mask = mgains >= 0.0
            negative_gain_mask = mgains < 0.0
            gain_items[positive_gain_mask] = np.maximum(0, np.minimum(mgains, remainder)[positive_gain_mask])
            gain_items[negative_gain_mask] = np.minimum(0, (mgains - remainder)[negative_gain_mask])
            
        scores = mask_scores(gain_items.sum(axis=0), seen_items_mask)
        i_best = scores.argmax()

        self.gm = self.gm + mgains[:, i_best]
        self.TOT = np.clip(self.gm, 0.0, None).sum()

        return i_best