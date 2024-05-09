import random
import numpy as np
from deap import algorithms, base, creator, tools
from termcolor import colored, cprint
from colorama import init

init()

class MagicCubeGA:
    def __init__(self, length):
        self.length = length
        self.magic_constant = length * (length**2 + 1) // 2
        self.toolbox = self.create_toolbox()

    def create_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register("attr_int", random.randint, 1, self.length**2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, self.length**2)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=1, up=self.length**2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.evaluate_magic_cube)

        return toolbox

    def evaluate_magic_cube(self, individual):
        distance = 0
        for i in range(self.length):
            row_sum = sum(individual[i*self.length: (i+1)*self.length])
            col_sum = sum(individual[j*self.length + i] for j in range(self.length))
            distance += abs(row_sum - self.magic_constant) + abs(col_sum - self.magic_constant)
        main_diag_sum = sum(individual[i*self.length + i] for i in range(self.length))
        anti_diag_sum = sum(individual[i*self.length + (self.length - 1 - i)] for i in range(self.length))
        distance += abs(main_diag_sum - self.magic_constant) + abs(anti_diag_sum - self.magic_constant)
        return distance,

    def solve(self):
        population = self.toolbox.population(n=100)
        algorithms.eaSimple(population, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)
        best_individual = tools.selBest(population, k=1)[0]
        return best_individual

    def display_magic_cube(self, individual):
        magic_cube = np.array(individual).reshape((self.length, self.length))
        for i in range(self.length):
            for j in range(self.length):
                if magic_cube[i][j] % 2 == 0:
                    cprint('{:^5}'.format(magic_cube[i][j]), 'red', end=" ")
                else:
                    cprint('{:^5}'.format(magic_cube[i][j]), 'blue', end=" ")
            print()

if __name__ == "__main__":
    length = int(input("Enter the length of the magic cube's edge: "))  # تعیین طول مربع جادویی
    magic_cube_ga = MagicCubeGA(length)
    solution = magic_cube_ga.solve()
    magic_cube_ga.display_magic_cube(solution)
