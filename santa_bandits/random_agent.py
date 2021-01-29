import random
import jsonlines
import json

def random_agent(observation, configuration):

	arm = random.randint(0,99)
	stored_data = { 
		"current_arm": arm,
		"observation": observation,
		"configuration": configuration
	}
	with jsonlines.open('random_agent.jsonl', 'a') as writer:
		writer.write(stored_data)
	return arm