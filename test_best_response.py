import matching
import numpy as np
import pandas as pd
import os
import min_cost_flow
from scipy import stats

FAVORITE_COURSES = 2
GOOD_COURSES = 2
OKAY_COURSES = 19

student_options = ['Okay', 'Good', 'Favorite']

def fixed_subgraph_weight(weights, student_data, course_data, sis, ci):
    if any(pd.isna(weights[sis, ci])):
        return -10000
    fixed_matches = pd.DataFrame(data={'Student index': sis, 'Course index': [ci for _ in range(len(sis))]})
    graph = min_cost_flow.MatchingGraph(weights, student_data, course_data, fixed_matches)
    graph.solve()
    return -graph.flow.OptimalCost() / 100

def get_feasible_student_matches(student_data, course_data, student_scores, course_scores, si):

    copy = student_scores.copy()
    copy.iloc[si, :] = 0
    weights = matching.match_weights(student_data, copy, course_scores)

    ws = []
    for ci in range(len(course_scores.index)):
        ws.append(fixed_subgraph_weight(weights, student_data['Weight'], course_data[['Slots', 'Base weight', 'First weight']], [si], ci))
    ord_ws = sorted(ws)

    feasible = []
    scores = list(set(stats.zscore([0] * OKAY_COURSES + [1] * GOOD_COURSES + [2] * FAVORITE_COURSES)))
    scores.sort()
    for ci in range(len(course_scores.index)):
        # rank ci first and remaining courses in reverse order by subgraph weight
        if ws[ci] + scores[2] >= ord_ws[FAVORITE_COURSES + GOOD_COURSES + OKAY_COURSES - 2] + scores[0] and ws[ci] + scores[2] >= ord_ws[FAVORITE_COURSES + GOOD_COURSES - 2] + scores[1] and ws[ci] >= ord_ws[EXCELLENT_COURSES - 2]:
            feasible.append(ci)
    return feasible

def rank_to_int(rank_str):
    return student_options.index(rank_str) if rank_str in student_options else -1

if __name__ == "__main__":
    path="br_test"
    student_data="inputs/student_data.csv"
    course_data="inputs/course_data.csv"
    fixed="inputs/fixed.csv"
    adjusted="inputs/adjusted.csv"
    output="outputs/"


    path = matching.validate_path_args(path, output)
    student_data = matching.read_student_data(path + student_data)
    course_data = matching.read_course_data(path + course_data)

    for student in student_data.index:
        if student not in course_data.columns.values:
            course_data[student] = np.nan
    for course in course_data.index:
        if course not in student_data.columns.values:
            student_data[course] = np.nan

    student_scores = matching.get_student_scores(student_data, course_data.index)
    course_scores = matching.get_course_scores(course_data, student_data.index)
    weights = matching.match_weights(student_data, student_scores, course_scores)

    fixed_matches = pd.DataFrame(columns=['Netid', 'Course', 'Student index', 'Course index'])
    graph = min_cost_flow.MatchingGraph(weights, student_data['Weight'], course_data[['Slots', 'Base weight', 'First weight']], fixed_matches)
    graph.solve()
    default_matching = sorted(graph.get_matching(fixed_matches))

    ci = 3
    copy = course_scores.copy()
    copy.iloc[ci, :] = 0
    weights = matching.match_weights(student_data, student_scores, copy)
    stn = len(student_scores.index)
    w = []
    for si1 in range(stn):
        if pd.notna(weights[si1, ci]):
            w.append([])
            for si2 in range(stn):
                if pd.notna(weights[si2, ci]):
                    if si1 == si2:
                        w[-1].append(0)
                    else:
                        w[-1].append(fixed_subgraph_weight(weights, student_data['Weight'], course_data[['Slots', 'Base weight', 'First weight']], [si1, si2], ci))
    np.set_printoptions(threshold=np.inf)
    # print(np.array(w))


    changes = []
    for si, (netid, row) in enumerate(student_data.iterrows()):
        course = course_data.index[default_matching[si][1]]
        print("{} matched to {} ({})".format(netid, course, row[course]))
        print("Feasible: ")
        feasible = get_feasible_student_matches(student_data, course_data, student_scores, course_scores, si)
        feasible_str = ["{} ({})".format(course_data.index[ci], row[course_data.index[ci]]) for ci in feasible]
        print(", ".join(feasible_str))
        changes.append((rank_to_int(row[course]), max(rank_to_int(row[course_data.index[ci]]) for ci in feasible)))

    for s in range(-1, 3):
        total = sum(1 for (si, _) in changes if si == s)
        improves = sum(1 for (si, sf) in changes if si == s and sf > si)
        print("{} out of {} students with {} initial improved".format(improves, total, s))

