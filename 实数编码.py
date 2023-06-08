import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 初始化种群
def initial_population(population_size,down,up):
    population = []
    for i in range(population_size):
        indivi = random.uniform(down, up)
        population.append(indivi)
    return population

# 适应度函数
def fitness_function(x):
    # 计算目标函数值
    y = x * np.sin(10 * math.pi * x) + 2.0
    # 返回适应度（即目标函数值）
    return y


# 适应度分数
def compute_fitness(population):
    fitness_values = []
    for i in range(len(population)):
        y = fitness_function(population[i])
        fitness_values.append(y)
    return fitness_values

# 选择操作
def selection(population, fitness_values, num_parents):
    # 保留适应度非负的个体,where返回的是一个二维数组
    positive_fitness_indices = np.where(np.array(fitness_values) >= 0)[0]
    # 根据下标找出个体和他的适应度值
    population = [population[i] for i in positive_fitness_indices]
    fitness_values = [fitness_values[i] for i in positive_fitness_indices]
    # 计算适应度总和
    fitness_sum = sum(fitness_values)
    # 计算每个个体的选择概率，与适应度分数成正比
    probabilities = [fitness_value / fitness_sum for fitness_value in fitness_values]
    # 计算累积概率分布
    cumulative_probabilities = np.cumsum(probabilities)
    # 选择父代个体
    parents = []
    for i in range(num_parents):
        # 产生一个0到1之间的随机数
        rand_num = np.random.random()
        # 确定随机数出现在哪个个体的概率区域内
        for j in range(len(cumulative_probabilities)):
            # 当前随机数小于等于累积概率列表中的某个元素，就选择该元素对应的个体作为父代
            if rand_num <= cumulative_probabilities[j]:
                parents.append(population[j])  # 直接返回基因
                break
    return parents


# 单点变异
def mutation(population,down,up):
    mutation = []
    for i in range(len(population)):
        y = random.uniform(0.7, 1.3)*population[i]
        if y<down:
            y=down
        elif y>up:
            y=up
        mutation.append(y)
    return mutation

#判断相识度
def simi(x,y,down,up):
    sim=abs(x-y)/(up-down)
    return sim

# 交叉
def cross(population,down,up):
    cross = []
    for i in range(len(population)):
        j=i
        while(j<(len(population)-1) and simi(population[i],population[j],down,up)<=0.2):
            j+=1
        ran = random.random()
        cross.append(ran * population[i] + (1-ran) * population[j])
    return(cross)

if __name__ == '__main__':
    population_size = 25  # 种群数量
    num_generations = 50  # 迭代次数
    num_parents = 20  # 父代数量
    down=-1
    up=2
    # 初始化种群
    population = initial_population(population_size,down,up)
    # 迭代num_generations轮
    best_fitness = float('-inf')
    best_individual = None
    best_fitnesses = []
    for generation in range(num_generations):
        # 计算适应度分数
        fitness_values = compute_fitness(population)
        # 取出局部最大
        index_max=fitness_values.index(max(fitness_values))
        population_copy = copy.deepcopy(population)
        tem_max=population_copy[index_max]
        # 选择父代个体
        parents = selection(population, fitness_values, num_parents)
        # 交叉操作
        cross_parents = cross(parents,down,up)
        # 变异操作
        mutation_parents = mutation(cross_parents,down,up)
        # 这代种群适应度
        fitness_mutation=compute_fitness(mutation_parents)
        # 取局部最小
        index_min=fitness_mutation.index(min(fitness_mutation))
        mutation_parents[index_min]=tem_max
        # 得到新的种群
        population = mutation_parents
        # 记录每一代的最好的适应度和个体
        generation_best_fitness = fitness_values[index_max]
        generation_best_individual = tem_max
        # 输出最佳个体
        print(
            f"Generation {generation + 1} - generation fitness: {generation_best_fitness:.6f}, generation individual Decimal: {generation_best_individual:.6f}")
        best_fitnesses.append(generation_best_fitness)
        # 更新全局最优解
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = generation_best_individual
    # # 读取二进制编码结果
    # heredity1=[]
    # with open('heredity.txt', 'r') as file:
    #     for line in file:
    #         float_line=float(line.strip())
    #         heredity1.append(float_line)

    # # 绘制每次迭代的最佳适应度
    # plt.plot(heredity1, label='binary')
    plt.plot(best_fitnesses, label='decimal')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')
    # 显示图例和图形
    plt.legend()
    plt.show()
    