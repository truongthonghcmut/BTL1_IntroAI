from genetic_nono import *
import random as rd
import pandas as pd
import psutil
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

# Tạo ra các cá thể ngẫu nhiên là đáp án của nonogram
def createIndividuals(nonogram, creator, initial_population):
    rows, cols, row_blocks, col_blocks = getInfomation(nonogram)
    individuals = [] # List of individuals - Danh sách các cá thể
    for i in range(initial_population):
        genotype = defineGenotypeForEachIndividual(cols) # Định nghĩa kiểu gen
        decoded = decodeGenotype(rows, genotype) # Giải mã gene thành nonogram
        individual = getSolution(cols, decoded) # Lấy ra đáp án của nonogram thành bảng
        individual = creator(individual) # Tạo ra cá thể
        individuals.append(individual)
    return individuals

# Hàm fitness: đếm số hàng không khớp của cá thể sinh ra so với thông tin nonogram ban đầu
# Fitness = 0 -> Cá thể sinh ra là đáp án của nonogram
def fitnessFunction(individual, nonogram):
    rows, cols, row_blocks, col_blocks = getInfomation(nonogram)
    nonogram = getNonogramInfomationFromIndividual(rows, col_blocks, individual)
    solution_rows = getRowsFromNonogram(nonogram)
    result = sum(x != y for x, y in zip(rows, solution_rows))
    return (result,)

# Selection: lựa chọn cá thể
# Elitism: chọn ra các cá thế tinh tú (fitness gần nhất với 0) để luôn giữ lại
def elitismSelection(population, toolbox, crossover_probability, mutation_probability, number_of_generations, stats=None, hall_of_fame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['generation', 'num_of_individuals', 'memory_usage_MB'] + (stats.fields if stats else []) # Lưu trữ thông tin về các thế hệ
    # Đánh giá fitness cho các cá thể chưa có điểm fitness
    invalid_individual = [individual for individual in population if not individual.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individual)
    for individual, fit in zip(invalid_individual, fitnesses):
        individual.fitness.values = fit
    # Sử dụng Hall of Fame để lưu trữ các cá thể tốt nhất
    if hall_of_fame is None:
        raise ValueError("Error: Hall of Fame cannot be empty")
    hall_of_fame.update(population)
    hall_size = len(hall_of_fame.items) if hall_of_fame.items else 0
    # Ghi nhận thông tin về thế hệ đầu tiên
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)
    record = stats.compile(population) if stats else {}
    record["memory_usage_MB"] = memory_usage
    logbook.record(generation=0, num_of_individuals=len(invalid_individual), **record)
    # In thông tin quá trình tìm ra đap án của nonogram
    print('\n')
    if verbose: print(logbook.stream)
    # Vòng lặp qua các thế hệ
    for gen in range(1, number_of_generations + 1):
        # Lựa chọn các cá thể tốt nhất tạo thành thế hệ tiếp theo (descendant)
        descendant = toolbox.select(population, len(population) - hall_size)
        descendant = algorithms.varAnd(descendant, toolbox, crossover_probability, mutation_probability)
        # Thực hiện biến đổi cho các cá thể chưa có điểm fitness, sau đó gán ngược lại điểm fitness cho các cá thể đó
        invalid_individual = [individual for individual in descendant if not individual.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individual)
        for individual, fit in zip(invalid_individual, fitnesses):
            individual.fitness.values = fit
        # Cập nhật Hall of Fame
        descendant.extend(hall_of_fame.items)
        hall_of_fame.update(descendant)
        # Cập nhật thế hệ hiện tại
        population[:] = descendant
        # Ghi nhận thông tin về thế hệ hiện tại        
        memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)
        record = stats.compile(population) if stats else {}
        record["memory_usage_MB"] = memory_usage
        logbook.record(generation=gen, num_of_individuals=len(invalid_individual), **record)
        if verbose: print(logbook.stream)
        # Nếu fitness của cá thể tốt nhất bằng 0 thì dừng lại, đây chính là đáp án của nonogram
        if record.get("min_fitness") == 0: break

    return population, logbook

# Sử dụng phương pháp lai ghép 2 điểm
def twoPointsCrossover(individual1, individual2):
    size = min(len(individual1), len(individual2))
    first_index = rd.randint(1, size) # Chọn ra điểm cắt thứ nhất 
    second_index = rd.randint(1, size - 1) # Chọn ra điểm cắt thứ hai
    # Xử lý để đảm bảo vị trí cắt thứ nhất nhỏ hơn vị trí cắt thứ hai
    if second_index >= first_index: second_index += 1
    else: first_index, second_index = second_index, first_index
    # Lai ghép 2 cá thể
    individual1[first_index:second_index], individual2[first_index:second_index] = individual2[first_index:second_index], individual1[first_index:second_index]
    return individual1, individual2

# Đột biến cá thể bằng cách hoán đổi vị trí
# probability: xác suất đột biến
def mutation(individual, probability, nonogram):
    rows, cols, row_blocks, col_blocks = getInfomation(nonogram)
    size = len(individual)
    for i in range(size):
        if rd.random() < probability:
            index = rd.randint(0, size - 1)
            result1 = sum(x + y if idx == 0 else x + y + 1 for idx, (x, y) in enumerate(zip(individual[index], cols[i])))
            result2 = sum(x + y if idx == 0 else x + y + 1 for idx, (x, y) in enumerate(zip(individual[i], cols[index])))
            if len(individual[index]) == len(individual[i]) and result1 <= 9 and result2 <= 9: individual[i], individual[index] = individual[index], individual[i]
            else: individual[i].reverse()
    return individual,

