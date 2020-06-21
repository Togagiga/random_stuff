# simple genetic algorithm

'''
starts with 4 5-bit strings as the first generation (each string is a binary number)
fitness of each member of generation is evaluated f(member)
members with higher fitness have higher probability to be in survivors (next gen)
survivors are mated in order they are samples with mutation of 10%
loop starts over again
'''

# with current settings algorithm achieves a max of 31 for the function x**2 - x

import numpy as np
from collections import deque

def num2bin(num):
	return f'{num:05b}'

def bin2num(bin):
	return int(bin, 2)

def numpy2str(array):
	str_out = ""
	for i in array:
		str_out += str(i)
	return str_out

func = lambda x : x**2 - x     # ------------> function to find maximum of

def getFitness(data):
	fitness = []
	for item in data:
		num = bin2num(item)
		fitness.append(func(num))
	return fitness

def getSurvivalChance(data):
	fitness = np.array(getFitness(data))
	total = sum(fitness)
	return fitness/total

def getCumProbability(data):
	probabilities = getSurvivalChance(data)
	cum_probs = [0]
	for i in range(len(probabilities)-1):
		if i == 0:
			cum_probs.append(probabilities[i])
		else:
			cum_probs.append(probabilities[i]+cum_probs[-1])
	
	return cum_probs

def getNextGen(cumulative_probabilities, generation_size = 4):
	next_gen = []
	for i in range(generation_size):
		sample = np.random.rand()
		# print(f"Sample: {sample}")
		for i in range(len(cumulative_probabilities)):
			if sample < cumulative_probabilities[i]:
				next_gen.append(i-1)
				break
			elif sample > cumulative_probabilities[-1]:
				next_gen.append(len(cumulative_probabilities)-1)
				break

	return next_gen

def performMutate(child):
	child = np.ones(len(child), dtype = int)
	child[:np.random.randint(0,4)] = 0
	np.random.shuffle(child)
	child = numpy2str(child)
	return child

def mateGen(next_gen):
	children = []
	next_gen = deque(next_gen)
	# mutation_const means percentage of children that DONT mutate
	mutate = False

	for i in range(int(len(next_gen)*0.5)):
		tail_length = np.random.randint(0,4)
		mutation_gen = np.random.rand()
		if mutation_gen > mutation_const:       # check whether to mutate or not
			mutate = True

		if tail_length == 0:     # children = parents
			child1 = data[next_gen.popleft()]
			child2 = data[next_gen.popleft()]
			# print(f"child1: {child1}, child2: {child2}")
			if mutate == True:
				child1 = performMutate(child1)
				child2 = performMutate(child2)
			children.append(child1)
			children.append(child2)

		else:
			parent1 = data[next_gen.popleft()]
			parent2 = data[next_gen.popleft()]

			child1 = parent1[:-tail_length] + parent2[-tail_length:]
			child2 = parent2[:-tail_length] + parent1[-tail_length:]
			# print(f"child1: {child1}, child2: {child2}")

			if mutate == True:
				child1 = performMutate(child1)
				child2 = performMutate(child2)
			children.append(child1)
			children.append(child2)

	return children

def runIteration(data):
	cum_probs = getCumProbability(data)
	next_gen = getNextGen(cum_probs)
	print(f"Next_Generation: {next_gen}")
	children = mateGen(next_gen)
	return children





data = ["01100", "11001", "01000", "10011"]     # starting generation

print(f"Function Values: {getFitness(data)}")
print(f"Probabilities: {getSurvivalChance(data)}")
# print(f"Cumulative Probabilities: {getCumProbability(data)}")

generations = 50
mutation_const = 0.9
for gen in range(generations):

	data = runIteration(data)
	print(data)
	print([bin2num(item) for item in data])