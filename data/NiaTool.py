from pydantic import BaseModel

class NiaTool(BaseModel):
    tool_name: str
    tool_description: str
    tool_type: str  # Type can be 'pre_response' or 'post_response'
    tool_definition: object  # This can be a dict or any other structure that defines the tool's parameters

    def __init__(self, tool_name: str, tool_description: str, type: str, tool_definition: object):
        super().__init__(tool_name=tool_name, tool_description=tool_description, tool_type=type, tool_definition=tool_definition) 

    def __str__(self):
        return f"{self.tool_name}: {self.tool_description} (Type: {self.tool_type}) \n Tool Definition: {self.tool_definition}"