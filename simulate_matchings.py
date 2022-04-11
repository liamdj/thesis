from hashlib import new
import random
import numpy as np
import pandas as pd
from numpy.random import default_rng
rng = default_rng()
from min_cost_flow import MatchingGraph
from scipy.optimize import linear_sum_assignment

def generate_utilities(quality):
    vals = rng.exponential(quality, quality.size)
    return vals / np.sum(vals)

def generate_ordering(quality):
    return rng.choice(np.arange(quality.size), quality.size, replace=False, p=quality)

def get_fixed_weights(student_qualities, course_qualities, student_weights, course_weights, number):
    """ Returns the fixed weights from the perspective of the first student,
        assuming everyone is truthful
    """
    n, m = student_qualities.size, course_qualities.size

    # course_info = pd.DataFrame({'Course': np.arange(m), 'Slots': [1]*m, 'Base weight': [0]*m, 'First weight': [0]*m})

    weights = np.zeros((number, m), dtype=int)
    for t in range(number):
        match_weights = np.zeros((n, m))
        for s in range(1, n):
            match_weights[s, :] += student_weights[np.argsort(generate_ordering(course_qualities))]
        for c in range(m):
            match_weights[:, c] += course_weights[np.argsort(generate_ordering(student_qualities))]

        for c in range(m):
            # fixed_matches = pd.DataFrame({'Course': [c], 'Student index': [0], 'Course index': [c]})
            # graph = MatchingGraph(match_weights, np.zeros(n), course_info, fixed_matches)
            # graph.solve()
            fixed = match_weights.copy()
            fixed[1:, c] = -np.inf
            row, col = linear_sum_assignment(fixed, maximize=True)
            weights[t, c] = int(round(fixed[row, col].sum()))
   
    return weights

def match_results(ranking, fixed_weights, rank_weights):
    assert(np.all(np.sort(ranking) == np.arange(ranking.size)))
    arr = fixed_weights + rank_weights[ranking][np.newaxis, :]
    # randomly select among equal entries
    return np.argmax(rng.random(arr.shape) * (arr == np.max(arr, axis=1)[:, np.newaxis]), axis=1)

def avg_utility(utilities, matches):
    return np.sum(utilities[matches]) / matches.size

def default_ranking(utilities):
    # element i is index of the course with index i in ranked order list
    ranked_index = np.flip(np.argsort(utilities))
    # element i is the index of course i in ranked order list
    return np.argsort(ranked_index)

def reorder_ranking(init_ranking, rank_from, rank_to):
    index = np.argwhere(init_ranking == rank_from)[0]
    if rank_to > rank_from:
        ranking = np.where((init_ranking <= rank_to) & (init_ranking > rank_from), init_ranking - 1, init_ranking)
    else:
        ranking = np.where((init_ranking >= rank_to) & (init_ranking < rank_from), init_ranking + 1, init_ranking)
    ranking[index] += rank_to - rank_from
    return ranking

def try_reordering(match_utilities, fixed_weights, student_weights):
    init_ranking = default_ranking(match_utilities)
    m = init_ranking.size
    results = np.zeros((m, m))
    for old_rank in range(m):
        for new_rank in range(m):
            ranking = reorder_ranking(init_ranking, old_rank, new_rank)
            matches = match_results(ranking, fixed_weights, student_weights)
            results[old_rank, new_rank] = avg_utility(match_utilities, matches)
    return results



if __name__ == "__main__":
    student_qualities = rng.random(10)
    student_qualities /= np.sum(student_qualities)
    course_qualities = rng.random(10)
    course_qualities /= np.sum(course_qualities)
    fixed = get_fixed_weights(student_qualities, course_qualities, np.arange(10, 0, -1), np.arange(10, 0, -1), 100)
    print(course_qualities, np.mean(fixed, axis=0))
