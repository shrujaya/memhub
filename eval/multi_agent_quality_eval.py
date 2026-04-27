import asyncio
import os
import sys
import uuid
import logging
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Ensure project root is in path
sys.path.insert(0, ".")

from agents.team_config import build_team
from agents.interceptor import attach_interceptors_to_team
from agents.tools import store_memory, store_shared_finding, check_memhub_health
from autogen import OpenAIWrapper

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("multi_agent_eval")

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    expected: str
    actual: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)

class MultiAgentQualityEvaluator:
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        self.results: List[ScenarioResult] = []

    def _disable_tools(self, team):
        """Aggressively strip tools from all agents and the manager."""
        agents_to_scrub = [
            team.get("orchestrator"), 
            team.get("researcher"), 
            team.get("analyst"), 
            team.get("critic"),
            team.get("manager")
        ]
        for agent in agents_to_scrub:
            if not agent: continue
            if hasattr(agent, "llm_config") and agent.llm_config:
                # Scrub the main config dict
                config = dict(agent.llm_config)
                config.pop("tools", None)
                config.pop("functions", None)
                
                # Scrub the config_list items too
                if "config_list" in config:
                    new_list = []
                    for item in config["config_list"]:
                        new_item = dict(item)
                        if "base_url" in new_item:
                            new_item["base_url"] = str(new_item["base_url"])
                        new_item.pop("tools", None)
                        new_item.pop("functions", None)
                        new_list.append(new_item)
                    config["config_list"] = new_list
                
                agent.llm_config = config
                
                # RE-INITIALIZE CLIENT to force config changes (CRITICAL)
                if hasattr(agent, "client"):
                    agent.client = OpenAIWrapper(**agent.llm_config)

                # 2. Patch system prompt to remove mentions of tool calling
                if hasattr(agent, "system_message") and agent.system_message:
                    new_prompt = agent.system_message.replace("call `query_team_memory`", "rely on the 'MemHub Working Memory' context provided below")
                    new_prompt = new_prompt.replace("call `store_shared_finding`", "tag your finding with [REMEMBER: <fact>]")
                    new_prompt = new_prompt.replace("call `store_memory`", "tag your finding with [REMEMBER: <fact>]")
                    agent.update_system_message(new_prompt)

    async def run_scenario_1_hidden_fact(self):
        """Scenario 1: Hidden Fact Retrieval"""
        name = "Hidden Fact Retrieval"
        logger.info(f"--- Starting Scenario: {name} ---")
        
        project_code = f"PHOENIX-{uuid.uuid4().hex[:6].upper()}"
        agent_id = "eval-researcher-01"
        
        # Seed the memory
        store_memory(agent_id=agent_id, content=f"The Project ID is {project_code}.")
        
        task = f"Retrieve the Project ID for agent {agent_id} and state it clearly."
        team = build_team(task=task, model=self.model, max_round=5)
        self._disable_tools(team)
        
        # Attach interceptor to automate retrieval
        attach_interceptors_to_team(team, {"Orchestrator": "eval-orchestrator-01", "Researcher": agent_id})
        
        # Run chat without explicit tools to avoid Ollama tool-support issues.
        # The interceptor will handle context injection automatically.
        await team["orchestrator"].a_initiate_chat(team["manager"], message=task)
        
        # Extract last message
        final_reply = team["group_chat"].messages[-1]["content"]
        passed = project_code in final_reply
        
        res = ScenarioResult(
            name=name,
            passed=passed,
            expected=project_code,
            actual=final_reply,
            metrics={"recall_accuracy": 1.0 if passed else 0.0}
        )
        self.results.append(res)
        logger.info(f"Result: {'PASSED' if passed else 'FAILED'}")

    async def run_scenario_2_cross_agent(self):
        """Scenario 2: Cross-Agent Handover"""
        name = "Cross-Agent Handover"
        logger.info(f"--- Starting Scenario: {name} ---")
        
        ingredient = "Glow-in-the-dark Mushrooms"
        researcher_id = "eval-researcher-02"
        analyst_id = "eval-analyst-02"
        
        # Researcher stores shared finding
        store_shared_finding(agent_id=researcher_id, content=f"Research discovery: The secret ingredient is {ingredient}.")
        
        task = f"As the Analyst, find the secret ingredient discovered by the Researcher and suggest a recipe using it."
        team = build_team(task=task, model=self.model, max_round=8)
        self._disable_tools(team)
        
        attach_interceptors_to_team(team, {
            "Orchestrator": "eval-orchestrator-02",
            "Researcher": researcher_id,
            "Analyst": analyst_id
        })
        
        await team["orchestrator"].a_initiate_chat(team["manager"], message=task)
        
        final_reply = ""
        for msg in reversed(team["group_chat"].messages):
            if msg["name"] == "Analyst":
                final_reply = msg["content"]
                break
        
        passed = ingredient.lower() in final_reply.lower()
        
        res = ScenarioResult(
            name=name,
            passed=passed,
            expected=ingredient,
            actual=final_reply,
            metrics={"collaboration_score": 1.0 if passed else 0.0}
        )
        self.results.append(res)
        logger.info(f"Result: {'PASSED' if passed else 'FAILED'}")

    async def run_scenario_3_policy_pressure(self):
        """Scenario 3: Policy Pressure (Demotion Test)"""
        name = "Policy Pressure (Demotion)"
        logger.info(f"--- Starting Scenario: {name} ---")
        
        agent_id = "eval-pressure-01"
        key_fact = "The target deployment date is October 24th."
        
        # Store 15 verbose memories to trigger demotion (assuming threshold is 2000 chars)
        for i in range(15):
            verbose_text = f"Observation {i}: This is a very long and detailed observation about system behavior that takes up significant space in the working memory buffer. " * 5
            store_memory(agent_id=agent_id, content=verbose_text)
            
        # Store the key fact last
        store_memory(agent_id=agent_id, content=key_fact)
        
        task = f"Check your memory for the target deployment date and report it."
        team = build_team(task=task, model=self.model, max_round=5)
        self._disable_tools(team)
        
        attach_interceptors_to_team(team, {"Orchestrator": "eval-orchestrator-03", "Researcher": agent_id})
        
        await team["orchestrator"].a_initiate_chat(team["manager"], message=task)
        
        final_reply = team["group_chat"].messages[-1]["content"]
        passed = "October 24th" in final_reply
        
        res = ScenarioResult(
            name=name,
            passed=passed,
            expected="October 24th",
            actual=final_reply,
            metrics={"robustness_score": 1.0 if passed else 0.0}
        )
        self.results.append(res)
        logger.info(f"Result: {'PASSED' if passed else 'FAILED'}")

    async def run_scenario_4_namespace(self):
        """Scenario 4: Namespace Isolation"""
        name = "Namespace Isolation"
        logger.info(f"--- Starting Scenario: {name} ---")
        
        researcher_id = "eval-researcher-04"
        private_fact = "My personal password is 'admin123'."
        shared_fact = "The public API key is 'PUBLIC_KEY_2024'."
        
        store_memory(agent_id=researcher_id, content=private_fact, namespace="private")
        store_shared_finding(agent_id=researcher_id, content=shared_fact)
        
        task = "Analyst, check the shared team memory for any API keys and report them. Do NOT report any private passwords."
        team = build_team(task=task, model=self.model, max_round=5)
        self._disable_tools(team)
        
        attach_interceptors_to_team(team, {
            "Orchestrator": "eval-orchestrator-04",
            "Analyst": "eval-analyst-04"
        })
        
        await team["orchestrator"].a_initiate_chat(team["manager"], message=task)
        
        final_reply = team["group_chat"].messages[-1]["content"]
        
        leaked = "admin123" in final_reply
        found_shared = "PUBLIC_KEY_2024" in final_reply
        passed = found_shared and not leaked
        
        res = ScenarioResult(
            name=name,
            passed=passed,
            expected="Found shared, No leak",
            actual=final_reply,
            metrics={"security_score": 1.0 if passed else 0.0}
        )
        self.results.append(res)
        logger.info(f"Result: {'PASSED' if passed else 'FAILED'}")

    async def run_scenario_5_conflict(self):
        """Scenario 5: Conflict Resolution"""
        name = "Conflict Resolution"
        logger.info(f"--- Starting Scenario: {name} ---")
        
        agent_id = "eval-conflict-01"
        fact_old = "Current Status: PENDING"
        fact_new = "Current Status: APPROVED (Updated 1 minute ago)"
        
        store_memory(agent_id=agent_id, content=fact_old)
        await asyncio.sleep(1) # Ensure different timestamp
        store_memory(agent_id=agent_id, content=fact_new)
        
        task = "What is the current status of the project based on the most recent memory?"
        team = build_team(task=task, model=self.model, max_round=5)
        self._disable_tools(team)
        
        attach_interceptors_to_team(team, {"Orchestrator": "eval-orchestrator-05", "Analyst": agent_id})
        
        await team["orchestrator"].a_initiate_chat(team["manager"], message=task)
        
        final_reply = team["group_chat"].messages[-1]["content"]
        passed = "APPROVED" in final_reply and "PENDING" not in final_reply
        
        res = ScenarioResult(
            name=name,
            passed=passed,
            expected="APPROVED",
            actual=final_reply,
            metrics={"recency_score": 1.0 if passed else 0.0}
        )
        self.results.append(res)
        logger.info(f"Result: {'PASSED' if passed else 'FAILED'}")

    def print_summary(self):
        print("\n" + "="*50)
        print(f"{'SCENARIO':<30} {'RESULT':<10}")
        print("="*50)
        for r in self.results:
            status = "✓ PASSED" if r.passed else "✗ FAILED"
            print(f"{r.name:<30} {status:<10}")
        print("="*50)

async def main():
    # Check health first
    try:
        health = check_memhub_health()
        if health.get("status") != "ok":
            print("Error: MemHub server is not healthy. Please run 'docker compose up' first.")
            return
    except Exception as e:
        print(f"Error connecting to MemHub: {e}")
        return

    # Use a simpler team build that doesn't force tools if possible,
    # or just use the default team and rely on the interceptor.
    evaluator = MultiAgentQualityEvaluator()
    
    # We can override the model if needed, but llama3.1 is the default.
    # Note: We rely on the Interceptor (agents/interceptor.py) which works
    # even if the model doesn't support the OpenAI Tool Calling API.
    
    await evaluator.run_scenario_1_hidden_fact()
    await evaluator.run_scenario_2_cross_agent()
    await evaluator.run_scenario_3_policy_pressure()
    await evaluator.run_scenario_4_namespace()
    await evaluator.run_scenario_5_conflict()
    
    evaluator.print_summary()

if __name__ == "__main__":
    asyncio.run(main())
