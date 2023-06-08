import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# 初始化种群
def generate_initial_population(population_size,dimension,code_length):
    population = []
    for i in range(population_size):
        temporary = []
        for j in range(dimension):
            indivi = random.randint(0, code_length)
            temporary.append(bin(indivi))
            # 随机选择一个值，可以是0或1，将其添加到染色体中
        population.append(temporary)
        # 将染色体添加到种群中
    return population


# 从二进制编码转换为十进制数值
def binary_to_decimal(population,scope,code_length):
    decimal_population = []
    for i in range(len(population)):
        temporary = []
        for j in range(len(population[i])):
            indivi = int(population[i][j], 2) # 二进制转十进制
            indivi = indivi*(scope[2*j+1] - scope[2*j])/code_length + scope[2*j]
            indivi=scope[2*j+1] if indivi > scope[2*j+1] else scope[2*j] if indivi < scope[2*j] else indivi
            temporary.append(indivi)
        decimal_population.append(temporary)
    return decimal_population  # 返回所有染色体映射后的十进制数值列表


# 十进制数值转换为二进制编码
def decimal_to_binary(population,scope,code_length):
    bin_population = []
    for i in range(len(population)):
        temporary = []
        for j in range(len(population[i])):
            indivi=scope[2*j+1] if population[i][j] > scope[2*j+1] else scope[2*j] if population[i][j] < scope[2*j] else population[i][j]
            indivi = int((indivi-scope[2*j])/(scope[2*j+1] - scope[2*j])*code_length)
            indivi = bin(indivi)
            temporary.append(indivi)
        bin_population.append(temporary)
    return bin_population



# 适应度函数
def fitness_function(decimal_list):
    # 计算目标函数值
    x = decimal_list[0]
    y = x * np.sin(10 * math.pi * x) + 2.0
    # 返回适应度（即目标函数值）
    return y


# 适应度分数
def compute_fitness(decimal_population):
    fitness_values = []
    for decimal in decimal_population:
        y = fitness_function(decimal)
        fitness_values.append(y)    #要改
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
    parents = []
    # 选择父代个体
    for i in range(num_parents):
        # 产生一个0到1之间的随机数
        rand_num = random.random()
        # 确定随机数出现在哪个个体的概率区域内
        for j in range(len(cumulative_probabilities)):
            # 当前随机数小于等于累积概率列表中的某个元素，就选择该元素对应的个体作为父代
            if rand_num <= cumulative_probabilities[j]:
                parents.append(population[j])  # 直接返回个体
                break

    return parents


def single_point_crossover(parents, crossover_rate):
    offspring = []
    num_parents = len(parents)
    num_dimension = len(parents[0])

    for i in range(0, num_parents - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        offspring1 = []
        offspring2 = []
        for j in range(num_dimension):
            bin1 = parent1[j]
            bin2 = parent2[j]
            indivi1=bin1[bin1.index('b')+1:]
            indivi2=bin2[bin2.index('b')+1:]
            
            num_bits1 = len(indivi1)
            num_bits2 = len(indivi2)
            crossover_point1 = random.randint(1, num_bits1)
            crossover_point2 = random.randint(1, num_bits2)
            remaining_point1 = crossover_point1-num_bits1
            remaining_point2 = crossover_point1-num_bits2 
            # 根据交叉率进行交叉操作
            if random.random() < crossover_rate:
                # 生成新的后代
                indivi1_tem = '0b'+indivi1[:crossover_point1] + indivi2[remaining_point1:].zfill(-remaining_point1)
                indivi2_tem = '0b'+indivi2[:crossover_point2] + indivi1[remaining_point2:].zfill(-remaining_point2)
                
            else:
                # 如果不交叉，则直接将父代作为后代
                indivi1_tem = bin1
                indivi2_tem = bin2

            offspring1.append(indivi1_tem)
            offspring2.append(indivi2_tem)

        # 将后代添加到列表中
        offspring.append(offspring1)
        offspring.append(offspring2)

    return offspring


# 变异操作
def mutation(offspring, mutation_rate):
    # 遍历每个后代
    for decimal_list in offspring:
        # 遍历每个后代的基因
        for indivi in decimal_list:
            # 判断是否进行变异操作 
            indivi = "0b".join(
    ["1" if random.random() < mutation_rate and indivi[i] == "0"
     else "0" if random.random() < mutation_rate and indivi[i] == "1"
     else indivi[i]
     for i in range(2,len(indivi))])
    # 返回变异后的后代
    return offspring


if __name__ == '__main__':
    population_size = 20  # 种群数量
    num_generations = 50  # 迭代次数
    num_parents = 20  # 父代数量
    crossover_rate = 0.9  # 交叉率
    mutation_rate = 0.1  # 变异率
    dimension = 1
    scope=[-1,2]  # 搜索空间边界，即每个基因x的取值范围为[-5, 5]
    code_length = 2000
    # 初始化种群
    population = generate_initial_population(population_size, dimension ,code_length)
    # 迭代num_generations轮
    best_fitness = float('-inf')
    best_individual = None
    best_fitnesses = []
    decimal_population=[]
    for generation in range(num_generations):
        # 二进制转换为十进制 
        decimal_population = binary_to_decimal(population, scope ,code_length)
        population = decimal_to_binary(decimal_population, scope ,code_length)
        fitness_values = compute_fitness(decimal_population)
        index_max=fitness_values.index(max(fitness_values))
        population_copy = copy.deepcopy(population)
        tem_max=population_copy[index_max]
        parents = selection(population, fitness_values, num_parents)
        # 交叉操作
        offspring = single_point_crossover(parents, crossover_rate)
        # 变异操作
        mutations = mutation(offspring, mutation_rate)
        decimal_mutations = binary_to_decimal(mutations, scope,code_length)
        # 取局部最小
        fitness_values_min = compute_fitness(decimal_mutations)
        index_min=fitness_values_min.index(min(fitness_values_min))
        mutations[index_min]=tem_max
        # 得到新的种群
        population = mutations
        # 记录每一代的最好的适应度和个体
        generation_best_fitness = fitness_values[index_max]
        decimal_best_individual = tem_max
        # 输出最佳个体的二进制编码和映射后的十进制值
        binary_best_individual = population[index_max]
        print(
            f"Generation {generation + 1} - generation fitness: {generation_best_fitness:.6f}, generation individual - Binary: {binary_best_individual}, Decimal: {decimal_best_individual}")
        best_fitnesses.append(generation_best_fitness)
        # 更新全局最优解
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = binary_best_individual

    # 绘制每次迭代的最佳适应度
    plt.plot(best_fitnesses, label='Best fitness per generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')
    # 显示图例和图形
    plt.legend()
    plt.show()
    