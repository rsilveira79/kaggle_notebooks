import random

def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)