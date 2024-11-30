import matplotlib.pyplot as plt
import numpy as np

# 데이터 입력
labels = ['1 Week', '2 Weeks', '3 Weeks', '4 Weeks']

cs_time = [0.46, 0.82, 1.08, 1.39]
cs_cost = [0, 0, 0, 0]
sa_time = [0.12, 2.1, 18, 166.15]
sa_cost = [0, 0, 0, 0.15]

cs_minus_time = [0.18, 0.42, 0.51, 0.59]
cs_minus_cost = [0, 0, 0, 0]
sa_minus_time = [0.06, 1.21, 11.98, 107.95]
sa_minus_cost = [0, 0, 0, 0.5]

cs_plus_time = [12.97, 18.81, 27.46, 30.25]
cs_plus_cost = [0, 0, 0, 0]
sa_plus_time = [3.07, 70.6, 215.36, 206.46]
sa_plus_cost = [0, 0.37, 5.75, 19.96]

x = np.arange(len(labels))

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 평균 실행 시간 그래프 (꺾은선 그래프, 색상 통일)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(labels, cs_time, marker='o', linestyle='-', color='b', label='CS')
ax.plot(labels, cs_minus_time, marker='o', linestyle='--', color='b', label='CS 완화')
ax.plot(labels, cs_plus_time, marker='o', linestyle=':', color='b', label='CS 강화')

ax.plot(labels, sa_time, marker='o', linestyle='-', color='r', label='SA')
ax.plot(labels, sa_minus_time, marker='o', linestyle='--', color='r', label='SA 완화')
ax.plot(labels, sa_plus_time, marker='o', linestyle=':', color='r', label='SA 강화')

ax.set_ylabel('Average Time (s)')
ax.set_title('조건별 평균 실행 시간')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig('average_execution_time.png')
plt.show()

# 평균 비용 그래프 (꺾은선 그래프, 색상 통일)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(labels, cs_cost, marker='o', linestyle='-', color='b', label='CS')
ax.plot(labels, cs_minus_cost, marker='o', linestyle='--', color='b', label='CS 완화')
ax.plot(labels, cs_plus_cost, marker='o', linestyle=':', color='b', label='CS 강화')

ax.plot(labels, sa_cost, marker='o', linestyle='-', color='r', label='SA')
ax.plot(labels, sa_minus_cost, marker='o', linestyle='--', color='r', label='SA 완화')
ax.plot(labels, sa_plus_cost, marker='o', linestyle=':', color='r', label='SA 강화')

ax.set_ylabel('Average Cost')
ax.set_title('조건별 평균 비용')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.savefig('average_cost.png')
plt.show()