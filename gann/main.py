import pygad

from .utils import calc_num_genes, calc_num_genes_lstm


class GANN:
    def __init__(self, fitness_func, on_generation):
        self.fitness_func = fitness_func
        self.on_generation = on_generation

    def init_ga_instance(
        self,
        in_size: int,
        out_size: int,
        layers: int,
        n_units: int,
        lstm: bool,
        num_generations: int,
        num_parents_mating: int,
        sol_per_pop: int,
        parent_selection_type: str = 'rank',
        crossover_type: str = 'uniform',
        crossover_probability: float = 0.9,
        mutation_type: str = 'random',
        mutation_probability: float = 0.1,
        mutation_percent_genes: int = 5,
    ):
        if lstm:
            num_genes = calc_num_genes_lstm(in_size, out_size, layers, n_units)
        else:
            num_genes = calc_num_genes(in_size, out_size, layers, n_units)

        self.ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            fitness_func=self.fitness_func,
            on_generation=self.on_generation,
            num_genes=num_genes,
            parent_selection_type=parent_selection_type,
            crossover_type=crossover_type,
            crossover_probability=crossover_probability,
            mutation_type=mutation_type,
            mutation_probability=mutation_probability,
            mutation_percent_genes=mutation_percent_genes,
        )

    def run(self):
        self.ga_instance.run()

    def best_solution(self):
        (
            solution,
            solution_fitness,
            solution_idx,
        ) = self.ga_instance.best_solution()
        return solution, solution_fitness, solution_idx

    def plot_fitness(self):
        self.ga_instance.plot_fitness()
