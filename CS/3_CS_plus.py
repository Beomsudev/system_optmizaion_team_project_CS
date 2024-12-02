import numpy as np
import random
import copy
import time
from scipy.stats import levy

# 비용 함수 (제약 조건 평가)
def evaluate_schedule_with_cost(schedule, period=7, fixed_schedule=None):
    f1, f2, f3 = 0, 0, 0
    total_days = period
    if fixed_schedule is not None:
        # 고정된 스케줄과 현재 스케줄을 결합하여 평가할 전체 스케줄을 생성
        schedule = [fixed + nurse for fixed, nurse in zip(fixed_schedule, schedule)]
        total_days += len(fixed_schedule[0])

    # 제약 조건 1: 하루 근무 인원 조건 (모닝, 애프터눈, 나이트 근무 인원 범위 체크)
    for day in range(len(schedule[0])):
        shifts = [nurse[day] for nurse in schedule]
        morning, afternoon, night = shifts.count(1), shifts.count(2), shifts.count(3)
        # 강화된 범위 설정: 모닝: 5~6명, 애프터눈: 4~5명, 나이트: 2~3명
        if not (5 <= morning <= 6) or not (4 <= afternoon <= 5) or not (2 <= night <= 3):
            f1 += 1

    # 제약 조건 2: 금지된 패턴 확인 (특정 연속 근무 패턴 금지)
    for nurse in schedule:
        # 금지된 연속 근무 패턴 체크 (애프터눈 -> 모닝, 나이트 -> 모닝 등)
        for day in range(len(nurse) - 1):
            if (nurse[day], nurse[day + 1]) in [(2, 1), (3, 1), (3, 2)]:
                f2 += 1
                break
        # 3일 연속 나이트 근무 금지
        for day in range(len(nurse) - 2):
            if nurse[day] == nurse[day + 1] == nurse[day + 2] == 3:
                f2 += 1
                break

    # 제약 조건 3: 근무 횟수 조건 (최근 기간 동안 각 근무 유형의 횟수 체크)
    valid_morning = period // 7 * 2
    valid_afternoon = period // 7 * 2
    valid_night = period // 7
    valid_rest = period - (valid_morning + valid_afternoon + valid_night)

    for nurse in schedule:
        recent_schedule = nurse[-period:]
        morning_count = recent_schedule.count(1)
        afternoon_count = recent_schedule.count(2)
        night_count = recent_schedule.count(3)
        rest_count = recent_schedule.count(0)
        if not (valid_morning - 1 <= morning_count <= valid_morning + 1 and
                valid_afternoon - 1 <= afternoon_count <= valid_afternoon + 1 and
                valid_night - 1 <= night_count <= valid_night + 1 and
                valid_rest - 1 <= rest_count <= valid_rest + 1):
            f3 += 1

    # 총 비용 계산 (각 제약 조건 위반 횟수에 가중치를 곱하여 계산)
    total_cost = f1 * 5 + f2 * 5 + f3 * 1
    return total_cost


# 초기 스케줄 생성
def generate_initial_schedules(n, nurses=15, days=7):
    return [[[random.choice([0, 1, 2, 3]) for _ in range(days)] for _ in range(nurses)] for _ in range(n)]

# Levy Flight 기반 새 스케줄 생성 (기존 스케줄에서 무작위 변경 적용)
def levy_flight_schedule(base_schedule, beta=1.5, max_changes_factor=0.3):
    new_schedule = copy.deepcopy(base_schedule)
    nurses = len(base_schedule)
    days = len(base_schedule[0])

    # Levy 분포를 사용해 max_changes 동적으로 설정
    levy_sample = levy.rvs(size=1)[0]  # Levy 분포에서 값 샘플링
    max_changes = int(abs(levy_sample) % int(max_changes_factor * nurses * days)) + 1
    max_changes = min(max_changes, nurses * days)  # 간호사 * 일 수 제한

    # 무작위로 스케줄 변경 수행
    changes = random.randint(1, max_changes)
    for _ in range(changes):
        nurse_idx = random.randint(0, nurses - 1)
        day_idx = random.randint(0, days - 1)
        current_value = new_schedule[nurse_idx][day_idx]
        new_value = random.choice([x for x in [0, 1, 2, 3] if x != current_value])
        new_schedule[nurse_idx][day_idx] = new_value

    return new_schedule

# Cuckoo Search Algorithm
def cuckoo_search(n, max_generations, pa, nurses=15, days=7, fixed_schedule=None):
    nests = generate_initial_schedules(n, nurses, days)
    costs = [evaluate_schedule_with_cost(schedule, period=days, fixed_schedule=fixed_schedule) for schedule in nests]
    start_time = time.time()

    for generation in range(max_generations):
        for i in range(n):
            new_schedule = levy_flight_schedule(nests[i])
            new_cost = evaluate_schedule_with_cost(new_schedule, period=days, fixed_schedule=fixed_schedule)
            random_nest_idx = random.randint(0, n - 1)

            if new_cost < costs[random_nest_idx]:
                nests[random_nest_idx] = new_schedule
                costs[random_nest_idx] = new_cost
            elif new_cost == costs[random_nest_idx]:
                if random.random() < 0.5:
                    nests[random_nest_idx] = new_schedule
                    costs[random_nest_idx] = new_cost

        num_replace = int(pa * n)
        for _ in range(num_replace):
            worst_idx = np.argmax(costs)
            nests[worst_idx] = levy_flight_schedule(nests[worst_idx])
            costs[worst_idx] = evaluate_schedule_with_cost(nests[worst_idx], period=days, fixed_schedule=fixed_schedule)

        sorted_indices = np.argsort(costs)
        nests = [nests[idx] for idx in sorted_indices]
        costs = [costs[idx] for idx in sorted_indices]

        if costs[0] == 0:
            break

    end_time = time.time()
    exec_time = end_time - start_time
    return nests[0], costs[0], generation + 1, exec_time

# 실행
if __name__ == "__main__":
    periods = [7, 14, 21, 28]
    cycles = 100
    n = 10
    max_generations = 10**6
    pa = 0.0

    total_results = {period: {"costs": [], "times": []} for period in periods}

    for cycle in range(1, cycles + 1):
        print(f"\n==== Cycle {cycle} ====")
        fixed_schedule = None
        for period in periods:
            best_schedule, best_cost, _, exec_time = cuckoo_search(
                n, max_generations, pa, nurses=15, days=7 if period == 7 else 7, fixed_schedule=fixed_schedule
            )
            total_results[period]["costs"].append(best_cost)
            total_results[period]["times"].append(exec_time)

            if fixed_schedule is None:
                fixed_schedule = best_schedule
            else:
                fixed_schedule = [fixed + nurse for fixed, nurse in zip(fixed_schedule, best_schedule)]

            # 단일 배열로 출력
            print(f"Period: {period} days, Best Cost: {best_cost}, Execution Time: {exec_time:.2f} seconds")
            print("Final Schedule:")
            for nurse in fixed_schedule:
                print(nurse)

    print("\n==== Averages ====")
    for period in periods:
        avg_cost = sum(total_results[period]["costs"]) / cycles
        avg_time = sum(total_results[period]["times"]) / cycles
        print(f"Period: {period} days, Avg Cost: {avg_cost:.2f}, Avg Time: {avg_time:.2f} seconds")
