from genetic_algo import *
from genetic_nono import *
import random as rd
import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import time

# 10 x 10 - test khó, ít thông tin, và có thể nhiều hơn 1 lời giải cho nonogram này
input = {
    "rows": [[2], [2], [2], [2], [2], [2], [2], [2], [2], [1]],
    "cols": [[1], [2], [2], [2], [2], [2], [2], [2], [2], [2]],
    "initial_population": 1500,
    "mutation_individual_probability": 0.05
}
# Xác định FitnessMin là tìm cá thể có điểm fitness nhỏ nhất (gần với 0 nhất) cho bài toán tối ưu hoá
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Xác định kiểu cá thể là Individual dựa trên danh sách list và có thuộc tính là FitnessMin
creator.create("Individual", list, fitness=creator.FitnessMin)
start = time.time()
# Đăng ký các thành phần dành cho thuật toán di truyền
tb = base.Toolbox()
tb.register("population", createIndividuals, input, creator.Individual)
tb.register("evaluate", fitnessFunction, nonogram=input)
tb.register("mate", twoPointsCrossover)
tb.register("mutate", mutation, probability=input.get("mutation_individual_probability"), nonogram=input)
tb.register("select", tools.selTournament, tournsize=3)
# Khởi tạo quần thế và Hall of Fame
population = tb.population(initial_population=input.get("initial_population"))
hall_of_fame = tools.HallOfFame(1)
# Đăng ký các thông số thống kê cần thiết cho việc lưu trữ thông tin về thế hệ
stats = tools.Statistics(lambda individual: individual.fitness.values)
stats.register("average", np.mean)
stats.register("min_fitness", np.min)
stats.register("max_fitness", np.max)
# Chạy thuật toán
population, logbook = elitismSelection(population, tb, crossover_probability=0.5, mutation_probability=0.2, number_of_generations=1000, stats=stats, hall_of_fame=hall_of_fame, verbose=True)
end = time.time()
rows, cols, row_blocks, col_blocks = getInfomation(input)
# Ghi nhận Hall of Fame đầu tiên là kết quả tốt nhất
result = getNonogramInfomationFromIndividual(rows, col_blocks, hall_of_fame[0])
print("\nElapsed Time: ", round(end - start, 5), "seconds")
print("\nPredicted Nonogram Solution:\n")
for row in result:
    line = ''
    for s in row:
        if s: line += chr(9608)
        else: line += ' '
    print(line)
