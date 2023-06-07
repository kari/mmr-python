"""
weng11a
A Bayesian Approximation Method for Online Ranking
Algorithm 1
"""

import math
from functools import reduce

BETAS = (25/6)**2
KAPPA = 0.0001

def probs(teams):
    """ Calculates skill rating implied winning probabilities """
    k = len(teams)

    m = [sum([j[0] for j in i]) for i in teams]
    ss = [sum([j[1] for j in i]) for i in teams]

    phats = []
    for i in range(k):
        ss_i = ss[i]
        m_i = m[i]

        phats_i = []
        for q in [q for q in range(k) if q != i]:
            ss_q = ss[q]
            m_q = m[q]

            c_iq = math.sqrt(ss_i + ss_q + 2*BETAS)

            phat_iq = math.exp(m_i/c_iq)/(math.exp(m_i/c_iq)+math.exp(m_q/c_iq))

            phats_i.append(phat_iq)

        phat_i = reduce(lambda x, y: x*y, phats_i)
        phats.append(phat_i)

    phats = [x/sum(phats) for x in phats]

    return phats

def update(teams, r):
    """ Algorithm 1 Update rules using the Bradley-Terry model with full pair """
    k = len(teams)
    # gamma_q = 1.0/k

    m = [sum([j[0] for j in i]) for i in teams]
    ss = [sum([j[1] for j in i]) for i in teams]

    new_teams = []
    for i in range(k):
        Gamma_i = 0
        Delta_i = 0
        ss_i = ss[i]
        m_i = m[i]
        for q in [q for q in range(k) if q != i]:
            ss_q = ss[q]
            m_q = m[q]

            if r[q] > r[i]:
                s = 1.0
            elif r[q] < r[i]:
                s = 0.0
            else:
                s = 0.5

            c_iq = math.sqrt(ss_i + ss_q + 2*BETAS)
            c_qi = c_iq
            phat_iq = math.exp(m_i/c_iq)/(math.exp(m_i/c_iq)+math.exp(m_q/c_iq))
            phat_qi = math.exp(m_q/c_qi)/(math.exp(m_q/c_qi)+math.exp(m_i/c_qi)) #phat_qi = 1-phat_iq

            gamma_q = math.sqrt(ss_i)/c_iq

            delta_q = ss_i / c_iq * (s - phat_iq)
            eta_q = gamma_q * (ss_i/c_iq**2) * phat_iq*phat_qi

            Gamma_i += delta_q
            Delta_i += eta_q

        new_players = []
        for j in range(len(teams[i])):
            m_ij = teams[i][j][0]
            ss_ij = teams[i][j][1]

            m_ij = m_ij + ss_ij/ss_i * Gamma_i
            ss_ij = ss_ij * max(1-ss_ij/ss_i * Delta_i, KAPPA)

            new_players.append((m_ij, ss_ij))

        new_teams.append(new_players)

    return new_teams
