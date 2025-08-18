from typing import List, Dict, Any, Optional

class FunctionParameter:
    def __init__(self, name: str, type: str, description: str):
        self.name = name
        self.type = type
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "description": self.description
        }

class FunctionDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[FunctionParameter],
        required: List[str] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required or []

    def to_dict(self) -> Dict[str, Any]:
        properties = {}
        for param in self.parameters:
            properties[param.name] = param.to_dict()

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required
                }
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FunctionDefinition":
        function_data = data["function"]
        params_data = function_data["parameters"]["properties"]
        
        parameters = []
        for param_name, param_props in params_data.items():
            parameters.append(FunctionParameter(
                name=param_name,
                type=param_props["type"],
                description=param_props["description"]
            ))
        
        return cls(
            name=function_data["name"],
            description=function_data["description"],
            parameters=parameters,
            required=function_data["parameters"].get("required", [])
        )
    

# Example usage - creating a function definition for Azure Search
# Assuming use_case_list and use_case variables are defined elsewhere
use_case_list = ["orders", "products", "reviews", "analytics"]  # Example values
use_case = "orders, products, reviews, analytics"  # Example value

# Create function parameters
search_query_param = FunctionParameter(
    name="search_query",
    type="string", 
    description="The user query related to e-commerce orders, products, reviews, status, analytics etc, e.g. find all orders by Chris Miller, Summarize the reviews of product P015"
)

use_case_param = FunctionParameter(
    name="use_case",
    type="string",
    description=f"The actual use case of the user query, e.g. {use_case}"
)

get_extra_data_param = FunctionParameter(
    name="get_extra_data",
    type="boolean",
    description="If true, fetch the extra data from NIA Finolex Search Index. If false, fetch the data from the use case index"
)

# Create the function definition
azure_search_function = FunctionDefinition(
    name="get_data_from_azure_search",
    description="Fetch the e-commerce order related documents from Azure AI Search for the given user query",
    parameters=[search_query_param, use_case_param, get_extra_data_param],
    required=["search_query", "use_case", "get_extra_data"]
)

# Example of how to convert to dict
function_dict = azure_search_function.to_dict()