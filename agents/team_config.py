import autogen
import os
from dotenv import load_dotenv
load_dotenv()

config_list = [
    {
        "model": "llama3", # Change this to the model you have in Ollama (e.g., mistral, llama2, etc.)
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama", # Ollama doesn't require a real key for local use
        "price": [0.0, 0.0], # <--- Adding this will stop the warning
    }
]
llm_config = {"config_list": config_list, "temperature": 0}

# Agent 1: The "Identity" Holder
agent_a = autogen.AssistantAgent(
    name="Identity_Agent",
    system_message="""You hold the Name and ID data for users.
    DATA: { "id": "USR-123", "name": "Jane Doe" }
    When asked about a user, provide only the Name and ID. 
    Collaborate with the Contact_Agent to create a full profile.""",
    llm_config=llm_config,
)

# Agent 2: The "Contact" Holder (The one we evaluate for Accuracy)
agent_b = autogen.AssistantAgent(
    name="Contact_Agent",
    system_message="""You hold the Email and Phone data for users.
    DATA: { "email": "jane@example.com", "phone": "555-0199" }
    Your task is to message the Identity_Agent, get their data, 
    and output a SINGLE merged JSON object.
    ACCURACY RULE: The final JSON must contain all 4 fields: id, name, email, and phone.
    If any field is missing or modified, the task is a FAILURE.
    When you are completely finished, prefix your final JSON with the exact word: FINAL_MERGED_JSON""",
    llm_config=llm_config,
)

# Orchestrator
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10, # Give them enough rounds to talk!
    is_termination_msg=lambda x: "FINAL_MERGED_JSON" in x.get("content", ""),
    code_execution_config={"use_docker": False}
)

if __name__ == "__main__":
    print("--- Starting Agent Collaboration Test ---")

    # 1. Create a group chat so all agents can see the messages and interact
    groupchat = autogen.GroupChat(
        agents=[user_proxy, agent_a, agent_b],
        messages=[],
        max_round=10
    )
    
    # 2. Create the manager that orchestrates the group chat routing
    manager = autogen.GroupChatManager(
        groupchat=groupchat, 
        llm_config=llm_config
    )

    # 3. The Admin initiates the chat with the manager to kick off the task
    try:
        user_proxy.initiate_chat(
            manager,
            message="Contact_Agent, please coordinate with Identity_Agent to get their data and output the final merged JSON object."
        )
        print("\n✅ Execution finished.")
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")