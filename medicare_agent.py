from typing import Dict, Any, List
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from databricks_langchain import ChatDatabricks
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import pandas as pd
from pyspark.sql import functions as F

# Import your existing functions
from intent_extractor import parse_user_intent
from plan_extractor import query_plans, parse_copay

class AgentState(TypedDict):
    """State that flows through the agent"""
    user_input: str
    extracted_intent: Dict[str, Any]
    plans: List[Dict[str, Any]]
    formatted_response: str
    error: str

def extract_intent_node(state: AgentState) -> AgentState:
    """
    Node that extracts intent from user input
    """
    try:
        user_text = state["user_input"]
        intent_result = parse_user_intent({"text": user_text})
        
        state["extracted_intent"] = intent_result
        return state
    except Exception as e:
        state["error"] = f"Error extracting intent: {str(e)}"
        return state

def query_plans_node(state: AgentState) -> AgentState:
    """
    Node that queries plans based on extracted intent
    """
    try:
        if "error" in state and state["error"]:
            return state
            
        intent = state["extracted_intent"]
        
        # Enhanced plan querying with better filtering
        service = intent['service']
        state_name = intent['location'].get('state', 'CA')  # Default to CA if not specified
        max_copay = intent.get('max_copay')
        
        # Query the database
        df = spark.table("team5.team5.plan_benefit_summary_view") \
            .filter((F.col("category_name").like(f"%{service}%")) & 
                   (F.col("state") == state_name) & 
                   (F.col("network_description") == "In-Network"))
        
        # If max_copay is specified, filter by it
        if max_copay is not None:
            df = df.filter(F.col("copay") <= max_copay)
        
        # Get top 5 plans with additional details
        plans_df = df.select(
            "plan_name",
            "copay",
            "coinsurance", 
            "deductible",
            "category_name",
            "benefit_name"
        ).distinct().orderBy("copay", "plan_name").limit(5)
        
        # Convert to list of dictionaries
        plans_list = []
        for row in plans_df.collect():
            plans_list.append({
                "plan_name": row["plan_name"],
                "copay": row["copay"],
                "coinsurance": row["coinsurance"],
                "deductible": row["deductible"],
                "category_name": row["category_name"],
                "benefit_name": row["benefit_name"]
            })
        
        state["plans"] = plans_list
        return state
        
    except Exception as e:
        state["error"] = f"Error querying plans: {str(e)}"
        return state

def format_response_node(state: AgentState) -> AgentState:
    """
    Node that formats the final response with nice explanations
    """
    try:
        if "error" in state and state["error"]:
            state["formatted_response"] = f"I'm sorry, I encountered an error: {state['error']}"
            return state
        
        intent = state["extracted_intent"]
        plans = state["plans"]
        
        # Initialize LLM for response formatting
        llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")
        
        format_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful Medicare plan advisor. Given the user's healthcare need and the matching plans, 
            create a friendly, informative response that:
            1. Acknowledges their specific need
            2. Explains why these plans are good matches
            3. Highlights key benefits and costs
            4. Provides actionable next steps
            
            Make the response warm, professional, and easy to understand."""),
            ("human", """
            User's Need: {service} in {location}
            Budget Constraint: {max_copay}
            
            Matching Plans:
            {plans_info}
            
            Please create a comprehensive, friendly response explaining these options.
            """)
        ])
        
        format_chain = LLMChain(llm=llm, prompt=format_prompt)
        
        # Prepare plans information
        plans_info = ""
        if plans:
            for i, plan in enumerate(plans, 1):
                copay_str = f"${plan['copay']}" if plan['copay'] else "Varies"
                deductible_str = f"${plan['deductible']}" if plan['deductible'] else "None"
                coinsurance_str = f"{plan['coinsurance']}%" if plan['coinsurance'] else "None"
                
                plans_info += f"""
                {i}. {plan['plan_name']}
                   - Copay: {copay_str}
                   - Deductible: {deductible_str}
                   - Coinsurance: {coinsurance_str}
                   - Coverage: {plan['benefit_name']}
                """
        else:
            plans_info = "Unfortunately, no plans were found matching your specific criteria."
        
        # Format location string
        location = intent['location']
        location_str = f"{location.get('city', '')}, {location.get('state', '')}"
        
        # Generate formatted response
        max_copay_str = f"Maximum copay of ${intent['max_copay']}" if intent.get('max_copay') else "No specific budget mentioned"
        
        formatted_response = format_chain.run(
            service=intent['service'],
            location=location_str,
            max_copay=max_copay_str,
            plans_info=plans_info
        )
        
        state["formatted_response"] = formatted_response
        return state
        
    except Exception as e:
        state["error"] = f"Error formatting response: {str(e)}"
        state["formatted_response"] = f"I found some plans for you, but encountered an error formatting the response: {str(e)}"
        return state

def create_medicare_agent():
    """
    Create and return the LangGraph agent
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("extract_intent", extract_intent_node)
    workflow.add_node("query_plans", query_plans_node)
    workflow.add_node("format_response", format_response_node)
    
    # Define the flow
    workflow.set_entry_point("extract_intent")
    workflow.add_edge("extract_intent", "query_plans")
    workflow.add_edge("query_plans", "format_response")
    workflow.add_edge("format_response", END)
    
    # Compile the graph
    app = workflow.compile()
    return app

def run_medicare_agent(user_input: str) -> str:
    """
    Main function to run the Medicare plan finder agent
    
    Args:
        user_input: Natural language input from user (e.g., "I need a root canal in SF, California. Max copay 50")
    
    Returns:
        Formatted response with top 5 Medicare plans and explanations
    """
    agent = create_medicare_agent()
    
    # Initialize state
    initial_state = AgentState(
        user_input=user_input,
        extracted_intent={},
        plans=[],
        formatted_response="",
        error=""
    )
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    return final_state["formatted_response"]

# Example usage and testing
if __name__ == "__main__":
    # Test the agent
    test_queries = [
        "I need a root canal in SF, California. Max copay 50",
        "Looking for dental coverage in New York with budget under $30",
        "Need emergency care coverage in Los Angeles, budget is $75 max",
        "I want vision coverage in Miami, Florida. Can spend up to $40"
    ]
    
    print("🏥 Medicare Plan Finder Agent - Test Run\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}: {query}")
        print("-" * 50)
        try:
            response = run_medicare_agent(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")
        print("\n" + "="*80 + "\n") 