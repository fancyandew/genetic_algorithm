import random
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

class initial:       
    def __init__(self,population_size,dimension, scope):
        self.population=[]
        self.population_size=population_size
        self.dimension=dimension
        self.scope=scope
    def generate(self):
        pass

class Binary_initial(initial):
    def __init__(self, population_size, dimension, scope, code_length):
        super().__init__(population_size, dimension, scope)
        self.code_length = code_length
        self.map_range = 2 ** self.code_length - 1
    def generate(self):
        for i in range(self.population_size):
            temporary = []
            for j in range(self.dimension):
                indivi = random.randint(0, self.map_range)
                temporary.append(bin(indivi)[2:].zfill(self.code_length))
            self.population.append(temporary)
        return self.population
    @staticmethod
    def binary_to_decimal(population,scope,code_length):
        decimal_population = []
        for i in range(len(population)):
            temporary = []
            for j in range(len(population[i])):
                indivi = int(population[i][j], 2) 
                indivi = indivi*(scope[1] - scope[0])/(2 ** code_length - 1) + scope[0]
                # indivi=scope[1] if indivi > scope[1] else scope[0] if indivi < scope[0] else indivi
                temporary.append(indivi)
            decimal_population.append(temporary)
        return decimal_population 
    
class Decimal_initial(initial):
    def generate(self):
        for i in range(self.population_size):
            temporary = []
            for  j in range(self.dimension):
                indivi = random.uniform(self.scope[0], self.scope[1])
                temporary.append(indivi)
            self.population.append(temporary)
        return self.population
    @staticmethod
    def decimal_to_binary(population,scope,code_length):
        bin_population = []
        for i in range(len(population)):
            temporary = []
            for j in range(len(population[i])):
                # indivi=scope[1] if population[i][j] > scope[1] else scope[0] if population[i][j] < scope[0] else population[i][j]
                indivi = int((population[i][j]-scope[0])/(scope[1] - scope[0])*(2 ** code_length - 1))
                indivi = bin(indivi)[2:].zfill(code_length)
                temporary.append(indivi)
            bin_population.append(temporary)
        return bin_population
    
class InitialFactory:
    _initial = {
        "Binary": Binary_initial,
        "Decimal": Decimal_initial
    }

    @classmethod
    def create_initial(cls, initial_type, population_size, dimension, scope, *args):
        if initial_type not in cls._initial:
            raise ValueError(f"Invalid initial type: {initial_type}")
        creator = cls._initial[initial_type]
        return creator(population_size, dimension, scope, *args).generate()

class FitnessFactory:
    _fitness = {
        "1": "fitness1_1",
        "2": "fitness2_1",
        "3": "fitness5_1"
    }

    @staticmethod
    def fitness1_1(x_list):
        y = x_list[0] * np.sin(10 * math.pi * x_list[0]) + 2.0
        return y

    @staticmethod
    def fitness2_1(x_list):
        y = 3*(x_list[0]**2-x_list[1])**2
        return y

    @staticmethod
    def fitness5_1(x_list):
        y = x_list[0]**2 + 10**6*(x_list[1]**2+x_list[2]**2+x_list[3]**2+x_list[4]**2)
        return y
    
    @classmethod
    def create_fitness(cls, fitness_type, population):
        fitness_values = []
        if fitness_type not in cls._fitness:
            raise ValueError("Invalid fitness_type type")
        creator = getattr(cls, cls._fitness[fitness_type])
        for i in range(len(population)):
            y = creator(population[i])
            fitness_values.append(y)
        return fitness_values

class SelectionFactory:
    _selection = {
        "max": "selection_max",
        "min": "selection_min",
        "max_positive": "selection_positive"
    }

    @staticmethod
    def selection_positive(population, fitness_values, num_select, best_fitness):
        if len(population) < num_select:
            raise ValueError("The number of selected individuals cannot exceed the population size.")
        if len(fitness_values) != len(population):
            raise ValueError("The lengths of population and fitness_values must be equal.")
        fitness_min = min(fitness_values)
        fitness_max = max(fitness_values)
        positive_fitness_indices = np.where(np.array(fitness_values) >= 0)[0]
        # 根据下标找出个体和他的适应度值
        population_positive = [population[i] for i in positive_fitness_indices]
        fitness_positive = [fitness_values[i] for i in positive_fitness_indices]
        fitness_sum = sum(fitness_positive)
        print("fitness_sum",fitness_sum)
        probabilities = [fitness_value / fitness_sum for fitness_value in fitness_positive]
        cumulative_probabilities = np.cumsum(probabilities)
        index_max = np.argmax(fitness_values)
        best_fitness.append(fitness_max)
        print(f"generation fitness:{fitness_max:.6f},generation best individual:{population[index_max]}")
        selection = []
        for i in range(num_select):
            rand_num = random.random()
            for j in range(len(cumulative_probabilities)):
                if rand_num <= cumulative_probabilities[j]:
                    selection.append(population_positive[j])
                    break
        return selection,population[index_max]
    
    @staticmethod
    def selection_max(population, fitness_values, num_select, best_fitness):
        if len(fitness_values) != len(population):
            raise ValueError("The lengths of population and fitness_values must be equal.")
        fitness_min = min(fitness_values)
        fitness_max = max(fitness_values)
        fitness_positive = [(value - fitness_min + 0.0000001) for value in fitness_values]
        fitness_sum = sum(fitness_positive)
        # print("fitness_sum",fitness_sum)
        probabilities = [fitness_value / fitness_sum for fitness_value in fitness_positive]
        cumulative_probabilities = np.cumsum(probabilities)
        index_max = np.argmax(fitness_values)
        best_fitness.append(fitness_max)
        print(f"generation fitness:{fitness_max:.6f},generation best individual:{population[index_max]}")
        selection = []
        for i in range(num_select):
            rand_num = random.random()
            for j in range(len(cumulative_probabilities)):
                if rand_num <= cumulative_probabilities[j]:
                    selection.append(population[j])
                    break
        return selection,population[index_max]
    
    @staticmethod
    def selection_min(population, fitness_values, num_select, best_fitness):
        if len(population) < num_select:
            raise ValueError("The number of selected individuals cannot exceed the population size.")
        if len(fitness_values) != len(population):
            raise ValueError("The lengths of population and fitness_values must be equal.")
        fitness_min = min(fitness_values)
        fitness_max = max(fitness_values)
        fitness_positive = [fitness_max-value + 0.0000001 for value in fitness_values]
        fitness_sum = sum(fitness_positive)
        probabilities = [fitness_value / fitness_sum for fitness_value in fitness_positive]
        cumulative_probabilities = np.cumsum(probabilities)
        index_min = np.argmin(fitness_values)
        best_fitness.append(fitness_min)
        print(f"generation fitness:{fitness_min:.6f},generation best individual:{population[index_min]}")
        selection = []
        for i in range(num_select):
            rand_num = random.random()
            for j in range(len(cumulative_probabilities)):
                if rand_num <= cumulative_probabilities[j]:
                    selection.append(population[j])
                    break
        return selection,population[index_min]
    
    @classmethod
    def create_selection(cls, selection_type, population, fitness_values, num_select, best_fitnesses):
        selection = []
        if selection_type not in cls._selection:
            raise ValueError("Invalid selection_type type")
        creator = getattr(cls, cls._selection[selection_type])
        selection = creator(population, fitness_values, num_select, best_fitnesses)
        return selection
    
class MutationFactory:
    _mutation = {
        "Binary": "binary_mutation",
        "Decimal": "decimal_mutation"
    }

    @staticmethod
    def binary_mutation(cross, mutation_rate):
        mutation = []
        for indivi_list in cross:
            # 遍历每个后代的基因
            tem = []
            for indivi in indivi_list:
                # 判断是否进行变异操作 
                new_indivi = ""  # 新的个体字符串
                for i in range(len(indivi)):
                    ran_num =random.random()
                    if indivi[i] == "0" and ran_num < mutation_rate:  # 如果当前基因位为0且发生突变
                        new_indivi += "1"  # 将该位基因翻转为1
                    elif indivi[i] == "1" and ran_num < mutation_rate:  # 如果当前基因位为1且发生突变
                        new_indivi += "0"  # 将该位基因翻转为0
                    else:  # 如果当前基因位不需要发生突变
                        new_indivi += indivi[i]  # 将该位基因保持不变
                tem.append(new_indivi)
            mutation.append(tem)
        # 返回变异后的后代
        return mutation
    
    @staticmethod
    def decimal_mutation(cross, mutation_rate):
        mutation = []
        for indivi_list in cross:
            # 遍历每个后代的基因
            tem = []
            for indivi in indivi_list:
                # 判断是否进行变异操作 
                indivi = random.uniform(0.7, 1.3)*indivi if random.random() < mutation_rate else indivi
                tem.append(indivi)
            mutation.append(tem)
        # 返回变异后的后代
        return mutation
    
    @classmethod
    def create_mutation(cls, mutation_type, cross, mutation_rate):
        mutation = []
        if mutation_type not in cls._mutation:
            raise ValueError("Invalid mutation_type type")
        creator = getattr(cls, cls._mutation[mutation_type])
        mutation = creator(cross, mutation_rate)
        return mutation

class CrossFactory:
    _cross = {
        "0": "weighted_average",
        "1": "similar_cross",
        "2": "random_crossover"
    }

    @staticmethod
    def weighted_average(selection, cross_rate =0.8):
        cross = []
        num_selection = len(selection)
        num_dimension = len(selection[0])

        for i in range(0, num_selection - 1, 2):
            parent1 = selection[i]
            parent2 = selection[i + 1]
            cross1 = []
            cross2 = []
            for j in range(num_dimension):
                # 根据交叉率进行交叉操作
                if random.random() < cross_rate:
                    # 生成新的后代
                    ran1 = random.random()
                    ran2 = random.random()
                    indivi1_tem = ran1 * parent1[j] + (1-ran1) * parent2[j]
                    indivi2_tem = ran2 * parent1[j] + (1-ran2) * parent2[j]
                else:
                    # 如果不交叉，则直接将父代作为后代
                    indivi1_tem = parent1[j]
                    indivi2_tem = parent2[j]

                cross1.append(indivi1_tem)
                cross2.append(indivi2_tem)
            cross.append(cross1)
            cross.append(cross2)
        return cross
    
    @staticmethod
    def similar_cross(selection, generation, fitness_values):
        # 适应度排序
        def get_element_order(lst):
            sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
            order = [x[0] for x in sorted_lst]
            return order
        # 余弦相识度
        def simi(u,v):
            sim=np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            sim = 0 if sim == 1 else sim
            return sim

        if generation <=10:
            return CrossFactory.weighted_average(selection)
        else:
            order = get_element_order(fitness_values)
            order_score = [x+1 / len(order) for x in order]
            cross = []
            num_selection = len(selection)
            num_dimension = len(selection[0])

            for i in range(num_selection):
                j=i
                while(j<(num_selection - 1) and random.random() > simi(selection[i],selection[j]) * order_score[i] * order_score[j]):
                    j+=1
                parent1 = selection[i]
                parent2 = selection[j]
                cross1 = []
                cross2 = []
                for j in range(num_dimension):
                    # 生成新的后代
                    ran1 = random.random()
                    ran2 = random.random()
                    indivi1_tem = ran1 * parent1[j] + (1-ran1) * parent2[j]
                    indivi2_tem = ran2 * parent1[j] + (1-ran2) * parent2[j]
                    cross1.append(indivi1_tem)
                    cross2.append(indivi2_tem)
                cross.append(cross1)
                cross.append(cross2)
            return cross
            
    
    @staticmethod
    def random_crossover(selection, cross_rate =0.8):
        cross = []
        num_selection = len(selection)
        num_dimension = len(selection[0])

        for i in range(0, num_selection - 2, 3):
            parent1 = selection[i]
            parent2 = selection[i + 1]
            parent3 = selection[i + 2]
            cross1 = []
            cross2 = []
            cross3 = []
            for j in range(num_dimension):
                # 根据交叉率进行交叉操作
                if random.random() < cross_rate:
                    # 生成新的后代
                    average = (parent1[j] + parent2[j] + parent3[j])/3
                    indivi1_tem = (1 + random.random()) * (parent1[j] - average)
                    indivi2_tem = (1 + random.random()) * (parent2[j] - average)
                    indivi3_tem = (1 + random.random()) * (parent3[j] - average)
                else:
                    # 如果不交叉，则直接将父代作为后代
                    indivi1_tem = parent1[j]
                    indivi2_tem = parent2[j]
                    indivi3_tem = parent3[j]

                cross1.append(indivi1_tem)
                cross2.append(indivi2_tem)
                cross3.append(indivi3_tem)
            cross.append(cross1)
            cross.append(cross2)
            cross.append(cross3)
        return cross

    @classmethod
    def create_cross(cls, cross_type,selection,*args):
        cross = []
        if cross_type not in cls._cross:
            raise ValueError("Invalid mutation_type type")
        creator = getattr(cls, cls._cross[cross_type])
        if len(args) == 0:
            cross = creator(selection)
        else:
            cross = creator(selection, *args)
        return cross

if __name__ == '__main__':
    # population_size = int(input("请输入种群数量：")) # 种群数量
    # dimension = int(input("请输入维度：")) #维数 
    # down = int(input("请输入下限："))
    # up = int(input("请输入上限："))
    # scope=[down,up]
    # coding = input("编码方式：Binary：0  Decimal：1 请输入编码方式：")
    # # 使用三元表达式实现相同的功能
    # coding_type = "Binary" if coding == "0" else ("Decimal" if coding == "1" else None)
    # # 如果 coding_type 为 None，则抛出错误提示
    # if coding_type is None:
    #     raise ValueError("coding 输入错误，请输入 0 或 1")
    # if coding_type == "Binary":
    #     code_length = int(input("请输入编码长度："))
    # print('''
    # f(x) = x * sin(10πx) + 2.0  1维  1
    # f(x₀, x₁) = 3*(x₀² - x₁)²   2维  2
    # f(x₀, x₁, x₂, x₃, x₄) = x₀² + 10⁶(x₁² + x₂² + x₃² + x₄²)    5维 3''')
    # fitness_type = input("请在以上目标函数中选择,输入对应的数字：")
    # selection_type = input('''
    # 选择最大值：0
    # 选择最小值：1
    # 请输入选择方式：''')
    # selection_type = "max" if selection_type == "0" else ("min" if selection_type == "1" else None)

    population_size = 30 # 种群数量
    dimension = 2 #维数 
    down = -10
    up = 10
    scope=[down,up]
    # 使用三元表达式实现相同的功能
    coding_type = "Decimal"
    code_length = 14
    fitness_type = "2"
    selection_type = "max"

    # population_size = 100 # 种群数量
    # dimension = 5 #维数 
    # down = -100
    # up = 100
    # scope=[down,up]
    # # 使用三元表达式实现相同的功能
    # coding_type = "Decimal"
    # code_length = 15
    # fitness_type = "3"
    # selection_type = "min"

    num_select = 20 
    mutation_rate = 0.1
    num_generations = 50  # 迭代次数
    best_fitness = []
    # cross_type = input("请在交叉算子中选择:\n加权平均：0\n变异概率：1\n随机平均：2\n输入对应的数字：")

    # 初始化种群
    if coding_type == "Binary":
        population = InitialFactory.create_initial(coding_type,population_size,dimension,scope,code_length)
    else :
        population = InitialFactory.create_initial(coding_type,population_size,dimension,scope)
    
    population1 = copy.deepcopy(population)
    population2 = copy.deepcopy(population)
    best_fitness1 = []
    best_fitness2 = []
    
    # 迭代num_generations轮
    for generation in range(num_generations):
        # print(population)
        # 计算适应度分数
        if coding_type == "Binary":
            population_decimal = Binary_initial.binary_to_decimal(population,scope,code_length)
            fitness_values = FitnessFactory.create_fitness(fitness_type,population_decimal)
        else:
            population = [[scope[1] if individual > scope[1] else scope[0] if individual < scope[0] else individual for individual in individual_list ] for individual_list in population]
            fitness_values = FitnessFactory.create_fitness(fitness_type,population)

        print(f"Generation {generation + 1} ",end='-')
        # 选择父代个体
        selection , best_individual= SelectionFactory.create_selection(selection_type,population,fitness_values,num_select,best_fitness)
        # print("selection",selection)
        # 交叉操作
        fitness_values = FitnessFactory.create_fitness(fitness_type,selection)
        cross_type = "0"
        # cross = CrossFactory.create_cross(cross_type,selection,generation,fitness_values)
        cross = CrossFactory.create_cross(cross_type,selection)
        # print("cross",cross)
        # 变异操作
        mutation= MutationFactory.create_mutation(coding_type,cross,mutation_rate)
        # print("mutation",mutation)
        mutation.append(best_individual)
        # 得到新的种群
        population = mutation

    # 迭代num_generations轮
    for generation in range(num_generations):
        # print(population)
        # 计算适应度分数
        if coding_type == "Binary":
            population_decimal = Binary_initial.binary_to_decimal(population,scope,code_length)
            fitness_values = FitnessFactory.create_fitness(fitness_type,population_decimal)
        else:
            population1 = [[scope[1] if individual > scope[1] else scope[0] if individual < scope[0] else individual for individual in individual_list ] for individual_list in population1]
            fitness_values = FitnessFactory.create_fitness(fitness_type,population1)

        print(f"Generation {generation + 1} ",end='-')
        # 选择父代个体
        selection , best_individual= SelectionFactory.create_selection(selection_type,population1,fitness_values,num_select,best_fitness1)
        # print("selection",selection)
        # 交叉操作
        fitness_values = FitnessFactory.create_fitness(fitness_type,selection)
        cross_type = "1"
        cross = CrossFactory.create_cross(cross_type,selection,generation,fitness_values)
        # cross = CrossFactory.create_cross(cross_type,selection)
        # print("cross",cross)
        # 变异操作
        mutation= MutationFactory.create_mutation(coding_type,cross,mutation_rate)
        # print("mutation",mutation)
        mutation.append(best_individual)
        # 得到新的种群
        population1 = mutation

    # 迭代num_generations轮
    for generation in range(num_generations):
        # print(population)
        # 计算适应度分数
        if coding_type == "Binary":
            population_decimal = Binary_initial.binary_to_decimal(population,scope,code_length)
            fitness_values = FitnessFactory.create_fitness(fitness_type,population_decimal)
        else:
            population2 = [[scope[1] if individual > scope[1] else scope[0] if individual < scope[0] else individual for individual in individual_list ] for individual_list in population2]
            fitness_values = FitnessFactory.create_fitness(fitness_type,population2)

        print(f"Generation {generation + 1} ",end='-')
        # 选择父代个体
        selection , best_individual= SelectionFactory.create_selection(selection_type,population2,fitness_values,num_select,best_fitness2)
        # print("selection",selection)
        # 交叉操作
        fitness_values = FitnessFactory.create_fitness(fitness_type,selection)
        cross_type = "2"
        # cross = CrossFactory.create_cross(cross_type,selection,generation,fitness_values)
        cross = CrossFactory.create_cross(cross_type,selection)
        # print("cross",cross)
        # 变异操作
        mutation= MutationFactory.create_mutation(coding_type,cross,mutation_rate)
        # print("mutation",mutation)
        mutation.append(best_individual)
        # 得到新的种群
        population2 = mutation

    plt.plot(best_fitness, label = "加权平均")
    plt.plot(best_fitness1, label = "变异概率")
    plt.plot(best_fitness2, label = "随机平均")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.title('工厂模式-二进制编码')
    # plt.title('工厂模式-实数编码')
    # 显示图例和图形
    plt.legend()
    plt.show()
