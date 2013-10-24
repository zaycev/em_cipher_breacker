# HOW TO RUN:
#
#   $ python breaker.py cipher.data english.data
#
#   This will train EM with 128 iterations, printing LL on the each iteration
#   and then output the final decipher with the substitution table to STDOUT.

import sys
import math
import numpy as np
import collections
import scipy.misc
import itertools
import decimal

DTYPE=np.float128
DEBUG=False
LOG=True
SMOOTH=True
MAX_ITER=128
MAX_REST=0
GETLOG=np.log
LOGADD=np.logaddexp
LOGSUM=scipy.misc.logsumexp


np.seterr(all="raise")
np.set_printoptions(linewidth=1024, suppress=True, precision=128)

def GET_LOG(x):
    if x > 0:
        return np.log(x)
    return -np.inf

def log_add(x, y):
    if x == -np.inf:
        return y
    if y == -np.inf:
        return x
    if x - y > 16:
        return x
    if y - x > 16:
        return y
    # return np.logaddexp(x, y)
    if x > y:
        return x + np.log(DTYPE(1) + np.exp(y - x))
    if y > x:
        return y + np.log(DTYPE(1) + np.exp(x - y))

def log_sum(arr):
    s = -np.inf
    for x in arr:
        s = log_add(s, x)
    return s

GETLOG=GET_LOG

def bigrams(tokens):
    for i in xrange(0, len(tokens) - 1):
        yield (tokens[i], tokens[i + 1])


def BIGRAM2K(bigram):
    return bigram[0] + bigram[1]


def K2BIGRAM(bigram_k):
    return bigram_k[0], bigram_k[1]


def collect_bigrams(text, smooth=False):
    text = text[0:(len(text) - 1)]

    p_u = dict()
    p_b = dict()

    for l in text:
        if l not in p_u:
            p_u[l] = np.float128(0)
        p_u[l] += np.float128(1)

    for t1 in p_u.keys():
        for t2 in p_u.keys():
            bg = BIGRAM2K((t1, t2))
            p_b[bg] = DTYPE(0)

    total = DTYPE(0)
    for bigram in bigrams(text):
        p_b[BIGRAM2K(bigram)] += DTYPE(1)
        total += DTYPE(1)

    if smooth:
        for t1 in p_u.keys():
            for t2 in p_u.keys():
                p_b[BIGRAM2K((t1, t2))] += DTYPE(1)
                total += DTYPE(1)
    P_B = dict()
    P_U = dict()

    total_log = GETLOG(DTYPE(total))
    for bigram in p_b.iterkeys():
        if bigram in p_b:
            P_B[bigram] = GETLOG(p_b[bigram]) - total_log
        else:
            P_B[bigram] = -np.inf

    total_log = GETLOG(DTYPE(len(text)))
    for letter in p_u.iterkeys():
        if p_u[letter] > 0:
            P_U[letter] = GETLOG(p_u[letter]) - total_log
        else:
            P_U[letter] = GETLOG(p_u[letter])


    return P_B, P_U


def collect_fractions(counts):
    totals = dict()
    for c_l in counts.iterkeys():
        if c_l[1] not in totals:
            totals[c_l[1]] = -np.inf
        totals[c_l[1]] = LOGADD(totals[c_l[1]], counts[c_l])

    P_c = dict()
    for c_l in counts.iterkeys():
        P_c[c_l] = counts[c_l] - totals[c_l[1]]
    return P_c


def naive_em(cipher, P_u, P_b, max_iter):
    P_c = dict()
    U = list(P_u.keys())
    C = list(set(cipher))

    for l in P_u.iterkeys():
        for c in C:
            bg = BIGRAM2K((c, l))
            P_c[bg] = - GETLOG(DTYPE(len(C)))

    for i in xrange(0, max_iter):

        P_compl = dict()
        for compl in itertools.product(U, repeat=len(cipher)):
            p = P_u[compl[0]] + P_c[cipher[0] + compl[0]]
            i = 1
            for i in xrange(1, len(compl)):
                l1 = compl[i - 1]
                l2 = compl[i]
                c = cipher[i]
                p += P_b[l1 + l2] + P_c[c + l]

            P_compl["".join(compl)] = p

        Z = LOGSUM(P_compl.values())
        P_compl = {k:(p - Z) for k,p in P_compl.iteritems()}

        counts = dict()
        for compl, p_compl in P_compl.iteritems():
            for i in xrange(len(cipher)):
                cl = cipher[i] + compl[i]
                if cl not in counts:
                    counts[cl] = -np.inf
                counts[cl] = LOGADD(counts[cl], p_compl)

        P_c = collect_fractions(counts)

        if DEBUG:
            print "C", [(k, np.exp(p)) for k,p in counts.iteritems()]
            print "T", [(k, np.exp(p)) for k,p in totals.iteritems()]
            print "N", [(k, np.exp(p)) for k,p in P_c.iteritems()]
            print np.exp(Z)

    decoded = viterbi(cipher, U, P_u, P_b, P_c)
    print "".join(decoded)


def train_em(cipher, P_u, P_b, max_iter, log, random, restarts):

    P_c = dict()
    U = list(P_u.keys())
    C = list(set(cipher))

    for _ in xrange(restarts if random else 1):

        # if random:
        #     for l in U:
        #         inits = np.random.randint(1, 1000, len(C))
        #         isum = np.sum(inits)
        #         for i, c in enumerate(C):
        #             bg = BIGRAM2K((c, l))
        #             P_c[bg] = GETLOG(DTYPE(inits[i]) / DTYPE(isum))
        # else:
        for l in P_u.iterkeys():
            for c in C:
                bg = BIGRAM2K((c, l))
                P_c[bg] = - GETLOG(DTYPE(len(C) - 1))
        
        for l in U:
            for c in C:
                if c == l == " ":
                    P_c[c + l] = GETLOG(DTYPE(1) - np.power(DTYPE(0.1), 32) * len(cipher))
                elif c == " " or l == " ":
                    P_c[c + l] = GETLOG(np.power(DTYPE(0.1), 32))


        best_PC = P_c
        best_LL = -np.inf

        for i in xrange(max_iter):

            log_likelihood, counts = FB2(cipher, U, P_u, P_b, P_c)
            P_c = collect_fractions(counts)

            if DEBUG:
                print "C", [(k, np.exp(p)) for k,p in counts.iteritems()]
                print "T", [(k, np.exp(p)) for k,p in totals.iteritems()]
                print "N", [(k, np.exp(p)) for k,p in P_c.iteritems()]
                print np.exp(log_likelihood)

            if log_likelihood >= best_LL:
                best_PC = P_c
                best_LL = log_likelihood
            else:
                print "ERROR"
                break


    # for b in P_c.keys():
    #     if np.exp(P_c[b]) > 0.01:
    #         print b[0]," => ",b[1], "   ", np.exp(P_c[b])


    decoded = viterbi(cipher, U, P_u, P_b, best_PC)
    print "".join(decoded)



def FB2(cipher, U, P_u, P_b, P_c):

    n = len(U)
    m = len(cipher) * 2

    A = np.zeros((n, m), dtype=DTYPE)
    B = np.zeros((n, m), dtype=DTYPE)

    A.fill(-np.inf)
    B.fill(-np.inf)

    for i in xrange(n):
        l = U[i]
        A[i,0] = P_u[l]

    for j in xrange(1, m):
        for i in xrange(n):
            l = U[i]
            c = cipher[j / 2]
            if j % 2 == 1:
                A[i, j] = A[i, j - 1] + P_c[c + l]
            else:
                for k in xrange(n):
                    l2 = U[k]
                    b = l2 + l
                    if b in P_b and P_b[b] > -np.inf:
                        A_prev = A[k, j - 1]
                        P_trns = P_b[b]
                        A[i, j] = LOGADD(A[i, j], A_prev + P_trns)

    ALP = LOGSUM(A[:,-1])
    B[:,-1] = DTYPE(0)

    for jj in xrange(1, m):
        j = m - jj - 1
        c = cipher[(j + 1) / 2]
        for i in xrange(n):
            l = U[i]
            if jj % 2 == 1:
                B[i, j] = B[i, j + 1] + P_c[c + l]
            else:
                for k in xrange(n):
                    l2 = U[k]
                    b =  l + l2
                    if b in P_b and P_b[b] > -np.inf:
                        B_next = B[k, j + 1]
                        P_trns = P_b[b]
                        B[i, j] = LOGADD(B[i, j], B_next + P_trns)

    counts = dict()
    for j in xrange(len(cipher)):
        for i in xrange(n):
            l = U[i]
            c = cipher[j]
            cl = c + l
            Alpha = A[i, j * 2]
            Beta = B[i, j * 2 + 1]
            count = Alpha + Beta + P_c[cl] - ALP
            if cl in counts:
                counts[cl] = LOGADD(counts[cl], count)
            else:
                counts[cl] = count

    return ALP, counts

def viterbi(cipher, U, P_u, P_b, P_c):

    n = len(U)                  # number of tags
    m = len(cipher)             # number of words
    Q = np.zeros((n, m), dtype=DTYPE)
    Q.fill(-np.inf)
    BP = np.zeros((n, m), dtype=np.int)

    for j in xrange(n):
        l = U[j]
        c = cipher[0]
        Q[j, 0] = P_u[l] + P_c.get(BIGRAM2K((c, l)), -np.inf)


    for i in xrange(1, m):  # <- words (cipher letters)
        for j in xrange(n): # <- tags (letters)

            Q[j, i] = -np.inf
            BP[j, i] = 0
            BS = -np.inf

            for k in xrange(n):
                bg = BIGRAM2K((U[k], U[j]))
                if bg not in P_b:
                    continue

                P_tj_tk = P_b[bg]
                P_wi_tj = P_c[BIGRAM2K((cipher[i], U[j]))]
                Q_prev = Q[k, i - 1]

                r =  P_tj_tk + P_wi_tj + Q_prev


                if r > BS:
                    BS = r
                    BP[j, i] = k
                    Q[j, i] = r


    FB = 0
    FS = -np.inf

    for j in xrange(n):
        if Q[j, m - 1] > FS:
            FS = Q[j, m - 1]
            FB = j

    decoded = [U[FB]]


    current = FB
    i = 0
    for jj in xrange(2, m + 1):
        j = m - jj
        current = BP[current, j + 1]
        decoded.append(U[current])

    decoded.reverse()

    return decoded


cipher_data = open(sys.argv[1], "r").read().replace("\n", " ")
english_data = open(sys.argv[2], "r").read().replace("\n", "")
P_b, P_u  = collect_bigrams(english_data, SMOOTH, LOG)


model = train_em(cipher_data, P_u, P_b, max_iter=MAX_ITER, random=MAX_REST>0, restarts=MAX_REST)
exit()


# C = "AABAABABAAAA"
# U = "YX"
# P_U = {"X":     GETLOG(0.6),    "Y":    GETLOG(0.4)}
# P_B = {"XX":    GETLOG(0.6),    "XY":   GETLOG(0.4),   "YX":   GETLOG(0.9),   "YY":    GETLOG(0.1)}
# P_C = {"AX":    GETLOG(0.6),    "BX":   GETLOG(0.4),   "AY":   GETLOG(0.5),   "BY":    GETLOG(0.5)}
# print "NAIVE"
# naive_em(C, P_U, P_B, max_iter=MAX_ITER)
# print
