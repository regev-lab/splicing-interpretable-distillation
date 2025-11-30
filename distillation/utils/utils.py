from math import factorial


def num_combinations_with_replacement(n, k):
    """
    Compute the number of combinations of k elements from a set of n
    elements with replacement.

    This is given by the formula (n + k - 1) choose (n - 1)
    (follows from doing a stars and bars argument).
    """
    return int(factorial(n - 1 + k) / (factorial(n - 1) * factorial(k)))

def rank_position(n, x):
    """
    Given a set of n elements and a particular k-tuple (from n^k), 
    return the lexicographical rank among all k-tuples in the 
    lexicographical ordering of the k-wise Cartesian product of n.
    """
    return sum([val * n ** (len(x) - i - 1) for i, val in enumerate(x)])

def rank_combination_with_replacement(n, k, x):
    """
    Rank a combination x of k elements from a set of n elements with replacement
    in the lexicographical ordering of all possible combinations with replacement.

    Given a combination x = (x_0, x_1, ..., x_k) (assume all combinations are
    represented with elements in lexographical order), where x_i denotes
    the i-th element in the combination, we can compute the rank using the
    following algorithm:

    At x_0, count the number of combinations that start with (y, ...) where y is
        less than x_0.
    At x_1, count the number of combinations that start with (x_0, y, ...) where
        y is at least x_0 but less than x_1.
    ...
    At x_i, count the number of combinations that start with (x_0, ..., x_{i-1}, y, ...)
        where y is at least x_{i-1} but less than x_i.

    For any x_i, this can be computed by \\Sigma_{j=lower}^{x_i - 1}
        num_comb_with_replacement(n - j, k - i - 1)
    where j = x_{i-1} when i > 0 and 0 otherwise.

    Thus, the total formula is:
    \\Sigma_{i=0}^{k - 1} \\Sigma_{j=0}^{x_i - 1}
        num_comb_with_replacement(n - j, k - i - 1)
    """
    x = sorted(x)
    rank = 0
    for i in range(k):
        lower = 0 if i == 0 else x[i - 1]
        for j in range(lower, x[i]):
            rank += num_combinations_with_replacement(n - j, k - i - 1)
    return rank


def num_combinations_without_replacement(n, k):
    """
    Compute the number of combinations of k elements from a set of n
    elements without replacement.
    """
    return int(factorial(n) / (factorial(n) * factorial(n - k)))
