from src.task import *
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent

META = "llama3.2:3b"
MODEL_NAME = META

tools = [sentiment_analysis, emotion_analysis, ner_analysis]

SYS_PROMPT = """
Tu es un assistant qui ne répond jamais directement.
Tu dois uniquement :
1. Déterminer la tâche demandée par l’utilisateur.
2. Appeler le bon outil.
3. Retourner la réponse brute du tool sans rien ajouter.
Si aucun outil ne correspond, réponds : "Tâche non supportée".
"""

TASK_DICO = {
    "sentiment_analysis": 0,
    "emotion_analysis": 1,
    "ner_analysis": 2,
}

class Agent:
    def __init__(self, model_name = "llama3.2:3b", temperature=0):
        self.model_name = model_name
        self.llm = OllamaLLM(
            model=model_name,
            prompt=PromptTemplate.from_template(SYS_PROMPT),
            temperature=temperature
        )
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix="You are a helpful assistant that uses APIs to perform tasks.",
            suffix="Begin!\n\nQuestion: {input}\n{agent_scratchpad}",
            input_variables=["input", "agent_scratchpad"]
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=[tool.name for tool in tools]
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            return_intermediate_steps=True
        )
    def run(self, user_query):
        result = self.agent_executor.invoke(user_query)
        return result["output"], result["intermediate_steps"][0][0].tool

if __name__ == "__main__":
    agent = Agent()
    question = "What’s your assessment of the sentiment in this text: I don’t have a strong opinion about the new treatment."
    print(agent.run(question))
