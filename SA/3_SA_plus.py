import numpy as np
import random
import copy
import time


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
def generate_initial_schedule(nurses=15, days=7, fixed_schedule=None):
    if fixed_schedule:
        remaining_days = days - len(fixed_schedule[0])
        new_schedule = [
            [random.choice([0, 1, 2, 3]) for _ in range(remaining_days)]
            for _ in range(nurses)
        ]
        return [fixed + nurse for fixed, nurse in zip(fixed_schedule, new_schedule)]
    else:
        return [[random.choice([0, 1, 2, 3]) for _ in range(days)] for _ in range(nurses)]


# 스케줄 변형 함수
def modify_schedule(schedule):
    new_schedule = copy.deepcopy(schedule)
    nurses = len(schedule)
    days = len(schedule[0])

    nurse_idx = random.randint(0, nurses - 1)
    for _ in range(random.randint(1, 3)):  # 한 번에 최대 3일 수정
        day_idx = random.randint(0, days - 1)
        current_value = new_schedule[nurse_idx][day_idx]
        new_schedule[nurse_idx][day_idx] = random.choice([x for x in [0, 1, 2, 3] if x != current_value])

    return new_schedule


# Simulated Annealing Algorithm
def simulated_annealing(nurses=15, days=7, max_iterations=10000, initial_temp=1000, cooling_rate=0.99,
                        fixed_schedule=None):
    # 고정 스케줄이 있을 경우 결합
    current_schedule = generate_initial_schedule(nurses, days, fixed_schedule)
    current_cost = evaluate_schedule_with_cost(current_schedule, period=days, fixed_schedule=fixed_schedule)
    best_schedule = copy.deepcopy(current_schedule)
    best_cost = current_cost

    temperature = initial_temp
    start_time = time.time()

    for iteration in range(max_iterations):
        new_schedule = modify_schedule(current_schedule)
        new_cost = evaluate_schedule_with_cost(new_schedule, period=days, fixed_schedule=fixed_schedule)

        # 비용 비교
        if new_cost < current_cost:
            current_schedule = new_schedule
            current_cost = new_cost
        else:
            # 비용이 더 높아도 확률적으로 수용
            probability = np.exp((current_cost - new_cost) / temperature)
            if random.random() < probability:
                current_schedule = new_schedule
                current_cost = new_cost

        # 최적해 갱신
        if current_cost < best_cost:
            best_schedule = copy.deepcopy(current_schedule)
            best_cost = current_cost

        # 온도 감소
        temperature *= cooling_rate

        # 종료 조건
        if best_cost == 0:
            break

    end_time = time.time()
    exec_time = end_time - start_time
    return best_schedule, best_cost, iteration + 1, exec_time


# 실행
if __name__ == "__main__":
    periods = [7, 14, 21, 28]
    cycles = 100
    max_iterations = 10 ** 6

    fixed_schedule = None  # 고정 스케줄 초기화

    # 각 기간별 결과 저장 리스트
    period_results = {period: {'costs': [], 'times': []} for period in periods}

    for cycle in range(1, cycles + 1):
        print(f"\n==== Cycle {cycle} ====")
        for period in periods:
            if period == 7:
                initial_temp, cooling_rate = 9500, 0.95
            elif period == 14:
                initial_temp, cooling_rate = 14250, 0.96
            elif period == 21:
                initial_temp, cooling_rate = 15000, 0.97
            else:
                initial_temp, cooling_rate = 20000, 0.98

            print(f"\n--- Period: {period} days ---")

            # 매 사이클 시작 시 고정 스케줄 초기화
            if cycle > 1 and period == 7:
                fixed_schedule = None

            best_schedule, best_cost, iterations, exec_time = simulated_annealing(
                nurses=15, days=period, max_iterations=max_iterations,
                initial_temp=initial_temp, cooling_rate=cooling_rate, fixed_schedule=fixed_schedule
            )

            # 고정 스케줄 업데이트
            fixed_schedule = best_schedule if best_cost == 0 else None

            print(
                f"Period: {period} days, Best Cost: {best_cost}, Iterations: {iterations}, Execution Time: {exec_time:.2f} seconds")
            print(f"Best Schedule for {period} days:")
            for nurse in best_schedule:
                print(nurse)

            # 결과 저장
            period_results[period]['costs'].append(best_cost)
            period_results[period]['times'].append(exec_time)

    # 평균 결과 출력
    print("\n==== Averages ====")
    for period in periods:
        avg_cost = sum(period_results[period]['costs']) / len(period_results[period]['costs'])
        avg_time = sum(period_results[period]['times']) / len(period_results[period]['times'])
        print(f"Period: {period} days, Avg Cost: {avg_cost:.2f}, Avg Time: {avg_time:.2f} seconds")