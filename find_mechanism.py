import numpy as np

# matching, index -> match
PREF_LABEL = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]])
# matching, match -> index
PREF_POS = np.argsort(PREF_LABEL, axis=1)
# ordering -> matching
PREF_INDEX = np.array([
    [[-1, -1, -1], [-1, -1, 0], [-1, 1, -1]],
    [[-1, -1, 2], [-1, -1, -1], [3, -1, -1]],
    [[-1, 4, -1], [5, -1, -1], [-1, -1, -1]]
])

# preferences that do not move options above match
def legal_swaps(pref, match):
    ranked_below = PREF_LABEL[pref][PREF_POS[pref, match]:]
    return [p for p in range(6) if not np.any(PREF_POS[p, match] > PREF_POS[p, ranked_below])]

# preferences after applying permutations
def permute(prefs, pis):
    arr = PREF_LABEL[prefs][..., pis]
    return PREF_INDEX[tuple(np.swapaxes(arr, 0, -1))]

# list of preferences for each player that are monotone with respect to a matching
def monotone_prefs(prefs, matching):
    ret = []
    for i in range(3):
        ret.append(legal_swaps(prefs[i], PREF_LABEL[matching][i]))
    for i in range(3):
        ret.append(legal_swaps(prefs[3 + i], PREF_POS[matching][i]))
    return ret

# indexing for allowing a player to submit any preferences
def altered_prefs(prefs_idx, player):
    ret = [np.expand_dims(prefs_idx[i], axis=1) for i in range(len(prefs_idx))]
    ret[player] = np.expand_dims(np.arange(6), axis=0)
    return tuple(ret)

# A mechanism is a partially-determined mapping between preferences and matchings
class Mechanism:
    def __init__(self, mech=None):
        if mech:
            self.mapping = mech.mapping.copy()
        else:
            self.mapping = -1 * np.ones([6 for _ in range(6)], dtype=np.int8)

    def complete(self):
        return np.all(self.mapping >= 0)
    
    def guess_next(self, matching):
        # find first location without a current matching
        prefs = np.array(np.unravel_index(np.argmax(self.mapping == -1), self.mapping.shape))
        return self.set_permutations(prefs, matching)

    # enforces neutrality
    def set_permutations(self, prefs, matching):
        # first permute course indices, then student indices
        s_prefs_idx = np.transpose(permute(prefs[:3], PREF_LABEL))
        s_prefs_idx = np.swapaxes(s_prefs_idx[PREF_LABEL, :], 0, 1)
        c_prefs_idx = np.transpose(prefs[3:][PREF_LABEL])
        c_prefs_idx = np.swapaxes(permute(c_prefs_idx, PREF_LABEL), 0, 2)
        matchings_permute = permute([matching], PREF_LABEL)[:, 0]
        matchings_permute = permute(matchings_permute, PREF_POS)
        combined = np.vstack((s_prefs_idx, c_prefs_idx))
        return self.set_output(tuple(np.reshape(combined, (6, -1))), np.ravel(matchings_permute))

    # enforces monoticity
    def set_monotone(self, prefs, matching):
        options = monotone_prefs(prefs, matching)
        prefs_idx = np.meshgrid(*options, sparse=True)
        return self.set_output(tuple(prefs_idx), matching)

    # determines if matching is currently strategyproof at given pref sets
    def check_strategyproof(self, prefs_idx):
        original_matches = PREF_LABEL[self.mapping[prefs_idx]]
        for i in range(3):
            matchings = self.mapping[altered_prefs(prefs_idx, i)]
            altered_matches = PREF_LABEL[matchings][:, :, i]
            altered_ranks = PREF_POS[np.expand_dims(prefs_idx[i], 1), altered_matches]
            altered_ranks[matchings == -1] = 3
            original_ranks = PREF_POS[prefs_idx[i], original_matches[:, i]]
            if np.any(np.min(altered_ranks, axis=1) < original_ranks):
                return False
        original_matches = PREF_POS[self.mapping[prefs_idx]]
        for i in range(3):
            matchings = self.mapping[altered_prefs(prefs_idx, 3+i)]
            altered_matches = PREF_POS[matchings][:, :, i]
            altered_ranks = PREF_POS[np.expand_dims(prefs_idx[3+i], 1), altered_matches]
            altered_ranks[matchings == -1] = 3
            original_ranks = PREF_POS[prefs_idx[3+i], original_matches[:, i]]
            if np.any(np.min(altered_ranks, axis=1) < original_ranks):
                return False
        return True
    
    # adds outputs to mechanism and checks if any properties are violated
    def set_output(self, prefs_idx, matchings):
        if np.any((self.mapping[prefs_idx] >= 0) & (self.mapping[prefs_idx] != matchings)):
            return False
        else:
            self.mapping[prefs_idx] = matchings
            # return True
            return self.check_strategyproof(prefs_idx)

# backtracking to find a feasible mechanism
def find_mechanism():
    root = Mechanism()
    queue = [root]
    n = 0
    while len(queue) > 0:
        last = queue.pop()
        n += 1
        if last.complete():
            return last
        else:
            for m in range(6):
                new = Mechanism(last)
                if new.guess_next(m):
                    queue.append(new)
    print(n)

def guess_mechanism():
    mech = Mechanism()
    while not mech.complete():
        if not mech.guess_next(0):
            return
    return mech

if __name__ == "__main__":
    find_mechanism()