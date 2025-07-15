import logging
from src.task import *
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from typing import List, Optional, Dict, Any
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END, START

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MODEL_NAME = "llama3.2:3b"

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

tools = [sentiment_analysis, emotion_analysis, ner_analysis]

class AgentState(BaseModel):
    input: str
    tools_to_call: Optional[List[Dict[str, Any]]] = None
    results: List[str] = Field(default_factory=list)
    current_tool_index: int = 0
    output: Optional[str] = None

class Agent:
    def __init__(self, model_name=MODEL_NAME, temperature=0, with_logs=False):
        self.with_logs = with_logs
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            prompt=PromptTemplate.from_template(SYS_PROMPT),
        ).bind_tools(tools)
        self.__init_params()

    def __init_params(self):
        graph = StateGraph(AgentState)
        graph.add_node("decision_node", self.decision_node)
        graph.add_node("tool_node", self.tool_node)
        graph.add_node("result_node", self.result_node)
        graph.add_edge(START, "decision_node")
        graph.add_edge("result_node", END)
        graph.add_conditional_edges("decision_node", path=self.route_decision_to_next,
                                    path_map={"tool_node": "tool_node", END: END})
        graph.add_conditional_edges("tool_node", path=self.route_tool_to_next,
                                    path_map={"tool_node": "tool_node", "result_node": "result_node"})
        graph.add_conditional_edges("result_node", path=self.route_result_to_end, path_map={END: END})
        self.app = graph.compile()
        png_data = self.app.get_graph().draw_mermaid_png()
        with open("../graph.png", "wb") as f:
            f.write(png_data)

    def decision_node(self, state: AgentState) -> AgentState:
        try:
            action = self.llm.invoke(state.input)
            tools_to_call = action.tool_calls
            if self.with_logs: logger.debug(f"[decision_node] Outils choisis : {tools_to_call}")
            if not tools_to_call:
                state.output = "Tâche non supportée"
                return state
            else:
                state.tools_to_call = tools_to_call
        except Exception as e:
            if self.with_logs: logger.error(f"[decision_node] Erreur: {e}")
            state.output = "Erreur lors de la sélection des outils"
        return state

    def tool_node(self, state: AgentState) -> AgentState:
        try:
            pred_tool = state.tools_to_call[state.current_tool_index]
            tool_name = pred_tool['name']
            tool_text = pred_tool['args']['text']
            if self.with_logs: logger.debug(f'Outils choisis par le LLM: {tool_name}')
            for tool in tools:
                if tool.name.lower() == tool_name:
                    prediction = tool.run(tool_text)
                    state.results.append(f"{tool.name}: {prediction}")
                    break
        except Exception as e:
            if self.with_logs: logger.error(f"[tool_node] Erreur avec {tool_name}: {e}")
            state.results.append(f"{tool_name}: ERREUR ({e})")
        finally:
            state.current_tool_index += 1
        return state

    @staticmethod
    def result_node(state: AgentState) -> AgentState:
        state.output = ",".join(state.results)
        return state

    @staticmethod
    def route_decision_to_next(state: AgentState):
        if state.tools_to_call is not None:
            return "tool_node"
        return END

    @staticmethod
    def route_tool_to_next(state: AgentState):
        if state.current_tool_index < len(state.tools_to_call):
            return "tool_node"
        return  "result_node"

    @staticmethod
    def route_result_to_end(state: AgentState):
        return END

    def run(self, state: AgentState):
        if isinstance(state, AgentState):
            return self.app.invoke(state)
        elif isinstance(state, dict):
            return self.app.invoke(state)
        else:
            raise TypeError("state must be AgentState or dict")
