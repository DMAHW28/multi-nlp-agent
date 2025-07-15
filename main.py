from src.agent import Agent, AgentState
if __name__ == "__main__":
    agent = Agent()
    state = AgentState(input="What’s your assessment of the sentiment in this text: I don’t have a strong opinion about the new treatment.")
    print(agent.run(state))
