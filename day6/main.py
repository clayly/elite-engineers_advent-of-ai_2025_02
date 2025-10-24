#!/usr/bin/env python3
"""
Minimal LangGraph Multi-Agent System
Two agents: Agent One (analysis) → Agent Two (synthesis)
"""

import os
import sys
from typing import Dict, Any, TypedDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# State definition
class MultiAgentState(TypedDict):
    messages: list
    current_agent: str
    agent_one_results: Dict[str, Any]
    agent_two_results: Dict[str, Any]
    error: str


# Simple logging
def log(message: str):
    print(f"[LOG] {message}")


# Agent One: Simple analysis
def agent_one(state: MultiAgentState) -> MultiAgentState:
    log("Agent One: Starting analysis")

    try:
        # Initialize LLM with z.ai API
        llm = ChatOpenAI(
            model="glm-4.6",
            temperature=0.7,
            max_tokens=1000,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )

        # Get the input message
        if state["messages"]:
            input_text = state["messages"][-1].content if hasattr(state["messages"][-1], 'content') else str(state["messages"][-1])
        else:
            input_text = "No input provided"

        # Simple analysis prompt
        prompt = f"""Analyze this text and provide key insights:

Text: {input_text}

Please provide:
1. Main topics
2. Key entities
3. Important findings
4. Brief summary"""

        # Call LLM
        response = llm.invoke(prompt)

        # Store results
        results = {
            "analysis": response.content,
            "input_length": len(input_text),
            "agent": "AgentOne"
        }

        log("Agent One: Analysis completed successfully")

        return {
            **state,
            "current_agent": "agent_two",
            "agent_one_results": results,
            "error": ""
        }

    except Exception as e:
        error_msg = f"Agent One error: {str(e)}"
        log(error_msg)
        return {
            **state,
            "current_agent": "error",
            "error": error_msg
        }


# Agent Two: Simple synthesis
def agent_two(state: MultiAgentState) -> MultiAgentState:
    log("Agent Two: Starting synthesis")

    try:
        # Initialize LLM with z.ai API
        llm = ChatOpenAI(
            model="glm-4.6",
            temperature=0.5,
            max_tokens=1500,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_BASE_URL")
        )

        # Get Agent One results
        agent_one_results = state.get("agent_one_results", {})
        analysis = agent_one_results.get("analysis", "No analysis available")

        # Simple synthesis prompt
        prompt = f"""Based on this analysis, create a concise summary and recommendations:

Analysis: {analysis}

Please provide:
1. Executive summary
2. Key takeaways
3. Recommendations"""

        # Call LLM
        response = llm.invoke(prompt)

        # Store results
        results = {
            "synthesis": response.content,
            "based_on_analysis": analysis[:100] + "...",
            "agent": "AgentTwo"
        }

        log("Agent Two: Synthesis completed successfully")

        return {
            **state,
            "current_agent": "end",
            "agent_two_results": results,
            "error": ""
        }

    except Exception as e:
        error_msg = f"Agent Two error: {str(e)}"
        log(error_msg)
        return {
            **state,
            "current_agent": "error",
            "error": error_msg
        }


# Route function
def route_agent(state: MultiAgentState) -> str:
    current = state.get("current_agent", "agent_one")

    if state.get("error"):
        return "__end__"

    if current == "agent_one":
        return "agent_two"
    elif current == "agent_two":
        return "__end__"
    else:
        return "__end__"


# Error handler
def error_handler(state: MultiAgentState) -> MultiAgentState:
    log(f"Error occurred: {state.get('error', 'Unknown error')}")
    return {**state, "current_agent": "__end__"}


# Create the graph
def create_graph():
    log("Creating multiagent graph")

    # Initialize graph
    workflow = StateGraph(MultiAgentState)

    # Add nodes
    workflow.add_node("agent_one", agent_one)
    workflow.add_node("agent_two", agent_two)

    # Set entry point
    workflow.set_entry_point("agent_one")

    # Add simple linear flow
    workflow.add_edge("agent_one", "agent_two")
    workflow.add_edge("agent_two", "__end__")

    # Add memory
    memory = MemorySaver()

    # Compile graph
    app = workflow.compile(checkpointer=memory)

    log("Graph created successfully")
    return app


# Main function
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your text to analyze\"")
        sys.exit(1)

    input_text = sys.argv[1]
    log(f"Starting multiagent processing for: {input_text[:50]}...")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Create graph
    app = create_graph()

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=input_text)],
        "current_agent": "agent_one",
        "agent_one_results": {},
        "agent_two_results": {},
        "error": ""
    }

    try:
        # Run the graph
        config = {"configurable": {"thread_id": f"session-{hash(input_text) % 10000}"}}
        result = app.invoke(initial_state, config=config)

        # Display results
        print("\n" + "="*60)
        print("MULTIAGENT PROCESSING RESULTS")
        print("="*60)

        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            # Agent One results
            agent_one = result.get("agent_one_results", {})
            if agent_one:
                print("\nAgent One Results:")
                print(f"Analysis length: {agent_one.get('input_length', 0)} characters")
                analysis = agent_one.get('analysis', 'No analysis')
                print(f"Analysis: {analysis[:300]}{'...' if len(analysis) > 300 else ''}")

            # Agent Two results
            agent_two = result.get("agent_two_results", {})
            if agent_two:
                print("\nAgent Two Results:")
                synthesis = agent_two.get('synthesis', 'No synthesis')
                print(f"Synthesis: {synthesis}")

        print(f"\nProcessing complete!")
        print(f"Final agent: {result.get('current_agent', 'unknown')}")

    except Exception as e:
        print(f"System error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()