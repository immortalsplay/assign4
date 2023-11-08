import numpy as np
import random
import matplotlib.pyplot as plt
import time
# 随机生成10个城市
num_cities = 1000
cities = np.random.rand(num_cities, 2)
data = []

with open('assign1/cities.txt', 'r') as file:
# with open('cities.txt', 'r') as file:
    for line in file:
        values = line.strip().split(', ')
        if len(values) == 2:
            data.append([float(values[0]), float(values[1])])

shuffled_data = data.copy()  # 复制数据以避免修改原数组
random.shuffle(shuffled_data)  # 随机排列

cities = np.array(shuffled_data)  # 将排列后的数据赋值给 cities
# 计算两个路径之间的距离
def calculate_distance(route, cities):
    distance = 0
    for i in range(len(route) - 1):
        distance += np.linalg.norm(cities[route[i]] - cities[route[i+1]])
    distance += np.linalg.norm(cities[route[-1]] - cities[route[0]])
    return distance

# 使用索引表示（Index Representation）的交叉操作
def index_crossover(parent1, parent2):
    index1 = sorted(range(len(parent1)), key=lambda k: parent1[k])
    index2 = sorted(range(len(parent2)), key=lambda k: parent2[k])
    
    n = random.randint(1, len(parent1))
    for i in range(n):
        index1[i], index2[i] = index2[i], index1[i]
    
    child1 = sorted(range(len(index1)), key=lambda k: index1[k])
    child2 = sorted(range(len(index2)), key=lambda k: index2[k])
    
    return child1, child2

# 初始化种群
population_size = 50
population = [random.sample(range(num_cities), num_cities) for _ in range(population_size)]

# 设定参数
num_generations = 2000
mutation_rate = 0.2

# 初始化用于记录的变量
avg_fitness_list = []
best_route = None
best_fitness = float('-inf')

# 主循环
time_start = time.time() #timer
all_runs_costs_at_intervals=[]
n=5
interval = 100
for run in range(n):
    avg_fitness_list = []  # 用于存储单次运行的平均适应度数据
    interval_costs = []
    for generation in range(num_generations):
        # 评估适应度
        distances = [calculate_distance(ind, cities) for ind in population]
        fitness = [1 / (d + 1e-6) for d in distances]
        
        # 找到并可能更新最佳解
        current_best_route = max(population, key=lambda x: 1 / calculate_distance(x, cities))
        current_best_fitness = 1 / calculate_distance(current_best_route, cities)
        
        if current_best_fitness > best_fitness:
            best_route = current_best_route
            best_fitness = current_best_fitness

        # 计算并保存平均适应度
        avg_fitness = np.mean(best_fitness)
        avg_fitness_list.append(avg_fitness)

        if generation % interval == 0:
            interval_costs.append(avg_fitness)
        # 轮盘赌选择
        fitness = [1 / (d + 1e-6) for d in distances]
        total_fitness = sum(fitness)
        probs = [f / total_fitness for f in fitness]
        selected_indices = np.random.choice(range(population_size), size=(population_size//2) * 2, p=probs, replace=False)
        selected_parents = [population[i] for i in selected_indices]
        

        # 交叉
        children = []
        for i in range(0, len(selected_parents) - 1, 2):
            child1, child2 = index_crossover(selected_parents[i], selected_parents[i + 1])
            children.extend([child1, child2])

        # 变异
        for child in children:
            if random.random() < mutation_rate:
                a, b = random.sample(range(len(child)), 2)
                child[a], child[b] = child[b], child[a]
        
        # 生成新一代
        population = sorted(population, key=lambda x: calculate_distance(x, cities))[:population_size//2]
        population += children[:population_size//2]  # 确保 population 的大小是 population_size
        
        # 输出当前最优解
        best_route = min(population, key=lambda x: calculate_distance(x, cities))
    all_runs_costs_at_intervals.append(interval_costs)


time_end = time.time()
print("Time Cost："+str((time_end - time_start))+"s")
# print(f"Generation {generation}: Best distance = {calculate_distance(best_route, cities)}")
# 输出最终最优解
best_route = min(population, key=lambda x: calculate_distance(x, cities))
print("Best route:", best_route)
print("Best distance:", calculate_distance(best_route, cities))

all_runs_costs_at_intervals = np.array(all_runs_costs_at_intervals)

# 计算平均值和标准差
avg_costs = np.mean(all_runs_costs_at_intervals, axis=0)
std_costs = np.std(all_runs_costs_at_intervals, axis=0)

# 计算误差条范围
min_costs = avg_costs - (std_costs / (2 * np.sqrt(n)))
max_costs = avg_costs + (std_costs / (2 * np.sqrt(n)))

# 绘图
x1 = np.arange(0, len(avg_costs)) * interval  # 这里的间隔需要和你实际的间隔相匹配

plt.plot(x1, avg_costs, label='Average Cost')
plt.errorbar(x1, avg_costs, yerr=[avg_costs - min_costs, max_costs - avg_costs], fmt='o', label='Error bars')

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Average Cost Over Iterations with Error Bars')
plt.legend()
plt.show()

with open('GA_short_avg_costs.txt', 'w') as f:
    for param1, param2 in zip(avg_costs, std_costs):
        f.write(f"{param1} {param2}\n")


# str_avg_fitness_list = [str(f).strip() + '\n' for f in all_runs_costs_at_intervals]

# with open('GA_short_avg_costs.txt', 'w') as f:
#     f.writelines(str_avg_fitness_list)


# 画图表示最佳路径
plt.figure(2)
city_x, city_y = zip(*[cities[i] for i in best_route])
plt.scatter(city_x, city_y, c='red')
plt.plot(city_x + (city_x[0],), city_y + (city_y[0],), c='blue')
for i, txt in enumerate(best_route):
    plt.annotate(txt, (city_x[i], city_y[i]))

plt.title('Best Path Found')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.show()