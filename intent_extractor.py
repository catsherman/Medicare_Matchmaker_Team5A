from typing import Dict, Any
import re
import json
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from databricks_langchain import ChatDatabricks

def extract_json_from_markdown(response: str) -> str:
    """
    Extract JSON content from markdown code blocks.
    
    Args:
        response: Response string that may contain JSON wrapped in markdown
        
    Returns:
        Clean JSON string
    """
    # Remove markdown code block syntax
    # Pattern matches ```json ... ``` or ``` ... ```
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # If no code block found, return original response stripped
    return response.strip()


def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response that may contain JSON in markdown format.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Parsed dictionary with service, location, and constraints
    """
    try:
        # Extract JSON from markdown if present
        clean_json = extract_json_from_markdown(response)
        
        # Parse the JSON
        parsed = json.loads(clean_json)
        
        # Validate required fields
        assert "service" in parsed and "location" in parsed
        
        return {
            "service": parsed["service"],
            "location": parsed["location"],
            "max_copay": parsed.get("max_copay")
        }
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")
    except AssertionError:
        raise ValueError("Response missing required fields: 'service' and 'location'")
    except Exception as e:
        raise ValueError(f"Error parsing response: {e}")

# -----------------------------
# LLM Setup (you can parameterize this later)
# -----------------------------
llm = ChatDatabricks(endpoint="databricks-llama-4-maverick")

# -----------------------------
# Prompt Template for Intent Parsing
# -----------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an assistant that extracts structured search intents for insurance plans. 
     You will receive a user message containing their healthcare needs, location, and budget.
     Based on input, choose the best service the user describes from the below categories:
     CATEGORY_NAMES = [
        "Maximum out-of-pocket enrollee responsibility (does not include prescription drugs)",
        "Medicare Part B drugs",
        "Hearing",
        "Diagnostic procedures/lab services/imaging",
        "Preventive care",
        "Ground ambulance",
        "Comprehensive Dental",
        "Additional benefits and/or reduced cost-sharing for enrollees with certain health conditions?",
        "Foot care (podiatry services)",
        "Skilled Nursing Facility",
        "Mental health services",
        "Medical equipment/supplies",
        "Vision",
        "Inpatient hospital coverage",
        "Emergency care/Urgent care",
        "Diagnostic and Preventive Dental",
        "Health plan deductible",
        "Rehabilitation services",
        "Doctor visits",
        "Outpatient hospital coverage",
        "Wellness programs (e.g., fitness, nursing hotline)",
        "Transportation",
        "Other health plan deductibles?",
        "Optional supplemental benefits"
     ]
     Return a JSON object with: 
     `service` (string), `location` (object with city and state), and `max_copay` (number). 
     If copay is not mentioned, return null for max_copay.
     
     Remember just return json output"""),
    ("human", "{text}")
])

parse_intent_chain = LLMChain(llm=llm, prompt=prompt)

# -----------------------------
# Core Function for LangGraph Node
# -----------------------------
def parse_user_intent(input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given input: { "text": "I need a root canal in SF, California. Max copay 50" }
    Return: {
        "service": "root canal",
        "location": { "city": "San Francisco", "state": "CA" },
        "max_copay": 50.0
    }
    """
    user_text = input.get("text", "")
    response = parse_intent_chain.run(text=user_text)

    try:
        parsed = parse_llm_response(response)
        assert "service" in parsed and "location" in parsed
        return {
            "service": parsed["service"],
            "location": parsed["location"],
            "max_copay": parsed.get("max_copay")
        }
    except Exception:
        return response
    
if __name__ == "__main__":
    test_input = {"text": "I need a root canal in SF, California. Max copay 50"}
    result = parse_user_intent(test_input)
    print(result)
    print(type(result))

