import numpy as np

# Test 1. 딕셔너리 덮어쓰기
V = {'L1': 0.0, 'L2': 0.0}

cnt = 0
while True:
    t = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    delta = abs(t - V['L1'])
    V['L1'] = t

    t = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    delta = max(delta, abs(t - V['L2']))
    V['L2'] = t

    cnt += 1
    if delta < 0.0001:
        print(V)
        print("갱신 횟수:", cnt)
        break
# {'L1': -2.2493782177156936, 'L2': -2.7494201578106514}
# 갱신 횟수: 60

#---------------------------------------------------------
# Test 2. 재귀함수
def func_dp(V: dict, gamma=0.9, tolerance=0.0001, cnt=0, max_iter=100000):
    '''
    :param V: {L1: float, L2: float} 의 초기값 dictionary
    :param gamma: 할인율
    :param tolerance: 임계값
    :param cnt: 갱신 횟수
    :param max_iter: 최대 갱신 횟수

    :return: newV, cnt + 1, delta
    '''
    t1 = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    t2 = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])


    delta = max(abs(t1 - V['L1']), abs(t2 - V['L2']))

    if delta < tolerance:
        return {'L1': t1, 'L2': t2}, cnt + 1, delta

    return func_dp({'L1': t1, 'L2': t2}, gamma=gamma, tolerance=tolerance, cnt=cnt+1, max_iter=max_iter)


V0 = {'L1': 0.0, 'L2': 0.0}
V_star, iters, last_delta = func_dp(V0, gamma=0.9, tolerance=0.0001)
print(V_star, "갱신횟수:", iters)

# {'L1': -2.249167525908671, 'L2': -2.749167525908671}
# 갱신횟수: 76

#----------------------------------------------------------
# 3. 선형대수 (권장)
gamma = 0.9
P = np.array([[0.5, 0.5],
              [0.5, 0.5]], dtype=float)
R = np.array([0.0, -0.5], dtype=float)

V = np.linalg.solve(np.eye(2) - gamma * P, R)
V_dict = {'L1': float(V[0]), 'L2': float(V[1])}
print(V_dict)
