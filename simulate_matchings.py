from hashlib import new
import random
import numpy as np
import pandas as pd
from numpy.random import default_rng
rng = default_rng()
from min_cost_flow import MatchingGraph

def generate_utilities(quality):
    sorted_utils = np.flip(np.sort(rng.random(quality.size)))
    ordering = generate_ordering(quality)
    return sorted_utils[ordering]

def generate_ordering(quality):
    return rng.choice(np.arange(quality.size), quality.size, replace=False, p=quality)

def get_fixed_weights(student_qualities, course_qualities, student_weights, course_weights, number):
    """ Returns the fixed weights from the perspective of the first student,
        assuming everyone is truthful
    """
    n, m = student_qualities.size, course_qualities.size

    course_info = pd.DataFrame({'Course': np.arange(m), 'Slots': [1]*m, 'Base weight': [0]*m, 'First weight': [0]*m})

    weights = np.zeros((number, m))
    for t in range(number):
        match_weights = np.zeros((n, m))
        for s in range(1, n):
            match_weights[s, :] += student_weights[generate_ordering(course_qualities)]
        for c in range(m):
            match_weights[:, c] += course_weights[generate_ordering(student_qualities)]

        for c in range(m):
            fixed_matches = pd.DataFrame({'Course': [c], 'Student index': [0], 'Course index': [c]})
            graph = MatchingGraph(match_weights, np.zeros(n), course_info, fixed_matches)
            graph.solve()
            weights[t, c] = graph.graph_weight()
   
    return weights

def ranking_performance(utilities, ranking, fixed_weights, rank_weights):
    assert(np.all(np.sort(ranking) == np.arange(ranking.size)))

    matches = np.argmax(fixed_weights + rank_weights[ranking][np.newaxis, :], axis=1)
    return np.sum(utilities[matches]) / matches.size

def try_reordering(match_utilities, student_qualities, course_qualities, student_weights, course_weights, trials):
    fixed_weights = get_fixed_weights(student_qualities, course_qualities, student_weights, course_weights, trials)
    # element i is index of the course with index i in ranked order list
    ranked_index = np.flip(np.argsort(match_utilities))
    # element i is the index of course i in ranked order list
    init_ranking = np.argsort(ranked_index)

    m = init_ranking.size
    results = np.zeros((m, m))
    for old_rank in range(m):
        ci = ranked_index[old_rank]
        for new_rank in range(m):
            if new_rank > old_rank:
                ranking = np.where((init_ranking <= new_rank) & (init_ranking > old_rank), init_ranking - 1, init_ranking)
            else:
                ranking = np.where((init_ranking >= new_rank) & (init_ranking < old_rank), init_ranking + 1, init_ranking)
            ranking[ci] += new_rank - old_rank
            results[old_rank, new_rank] = ranking_performance(match_utilities, ranking, fixed_weights, student_weights)
    return results



if __name__ == "__main__":
    student_qualities = rng.random(10)
    student_qualities /= np.sum(student_qualities)
    course_qualities = rng.random(10)
    course_qualities /= np.sum(course_qualities)
    fixed = get_fixed_weights(student_qualities, course_qualities, np.arange(10, 0, -1), np.arange(10, 0, -1), 100)
    print(course_qualities, np.mean(fixed, axis=0))
