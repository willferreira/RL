"""
RL Assignment - Easy21
@author: William Ferreira

We now consider a simple value function approximator using coarse coding. Use a binary
feature vector φ(s, a) with 3 ∗ 6 ∗ 2 = 36 features. Each binary feature has a value
of 1 iff (s, a) lies within the cuboid of state-space corresponding to that feature,
and the action corresponding to that feature. The cuboids have the following
overlapping intervals:

dealer(s) = {[1, 4], [4, 7], [7, 10]}
player(s) = {[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]}
a = {hit, stick}

where
• dealer(s) is the value of the dealer’s first card (1–10)
• sum(s) is the sum of the player’s cards (1–21)

Repeat the Sarsa(λ) experiment from the previous section, but using linear value function
approximation Q(s, a) = φ(s, a)⊤θ. Use a constant exploration of ε = 0.05 and a constant
step-size of 0.01. Plot the mean-squared error against λ. For λ = 0 and λ = 1 only,
plot the learning curve of mean-squared error against episode number.

                                                                            15 marks
"""


class Easy21FunctionApprox(object):
    pass