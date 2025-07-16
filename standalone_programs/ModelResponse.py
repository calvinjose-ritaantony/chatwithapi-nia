from typing import List, Dict, Any, Optional, Union, Annotated
from pydantic import BaseModel, Field
import json

# Data structures for different visualization types
class TableData(BaseModel):
    """Data structure for table representation"""
    headers: List[str] = Field(..., description="Column headers for the table")
    rows: List[List[Any]] = Field(..., description="Table data rows")
    caption: Optional[str] = Field(None, description="Optional table caption/description")
    use_case_type: str = Field("GENERIC", description="Type of use case this table represents")
    
    # Use case specific data that can be used for further processing
    use_case_data: Optional[Dict[str, Any]] = Field(None, description="Use case specific structured data")

class GraphData(BaseModel):
    """Data structure for graph/chart representation"""
    chart_type: str = Field(..., description="Type of chart: 'bar', 'line', 'pie', 'scatter', etc.")
    title: str = Field(..., description="Title of the graph")
    x_axis_label: Optional[str] = Field(None, description="Label for X-axis")
    y_axis_label: Optional[str] = Field(None, description="Label for Y-axis")
    labels: List[str] = Field(..., description="Labels for data points/categories")
    datasets: List[Dict[str, Any]] = Field(..., description="Data series with format {label, data, backgroundColor, etc}")
    description: Optional[str] = Field(None, description="Optional description of what the graph represents")
    
    # Additional configuration options for charts
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional chart configuration options")

class ImageData(BaseModel):
    """Data structure for image generation requests"""
    prompt: str = Field(..., description="Prompt describing the image to generate")
    alt_text: str = Field(..., description="Accessibility alt text for the image")
    width: int = Field(1024, description="Desired width of the image")
    height: int = Field(1024, description="Desired height of the image")
    style: Optional[str] = Field(None, description="Style directive for image generation")
    description: Optional[str] = Field(None, description="Description of the image's purpose in the response")

class PDFDocumentData(BaseModel):
    """Data structure for PDF document generation"""
    title: str = Field(..., description="Document title")
    subtitle: Optional[str] = Field(None, description="Document subtitle")
    sections: List[Dict[str, Any]] = Field(..., description="Document sections with headings and content")
    include_table_of_contents: bool = Field(True, description="Whether to include a table of contents")
    include_header_footer: bool = Field(True, description="Whether to include headers and footers")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Document metadata like author, subject, etc.")

class CodeData(BaseModel):
    """Data structure for code snippets"""
    language: str = Field(..., description="Programming language of the code")
    code: str = Field(..., description="The actual code content")
    filename: Optional[str] = Field(None, description="Suggested filename if code should be saved")
    description: Optional[str] = Field(None, description="Description or purpose of the code")

class UseCaseOutputDataUnion(BaseModel):
    """Union of all possible use case output data types"""
    use_case_type: str = Field(..., description="The use case type this data corresponds to")
    data: Dict[str, Any] = Field(..., description="The actual structured data for the use case")

class ModelResponse(BaseModel):
    """
    Comprehensive model response structure supporting multiple output formats.
    The model can intelligently determine which output formats to use based on the query.
    """
    # Core response text - always included
    model_response: str = Field(
        ..., 
        description="Main textual response from the model"
    )
    
    # Metadata and processing information
    total_tokens: int = Field(
        0, 
        description="Total tokens used in generating the response"
    )
    
    follow_up_questions: List[str] = Field(
        default_factory=list, 
        description="Suggested follow-up questions for the user"
    )
    
    reasoning: str = Field(
        "", 
        description="Model's reasoning process in generating this response (not shown to user)"
    )
    
    error_message: str = Field(
        "", 
        description="Error message if something went wrong during processing"
    )
    
    # Output format flags - indicate which additional data structures are included
    has_table: bool = Field(
        False, 
        description="Flag indicating if table data is included in the response"
    )
    
    has_graph: bool = Field(
        False, 
        description="Flag indicating if graph data is included in the response"
    )
    
    has_code: bool = Field(
        False, 
        description="Flag indicating if code snippets are included in the response"
    )
    
    has_image: bool = Field(
        False, 
        description="Flag indicating if image generation requests are included"
    )
    
    has_pdf: bool = Field(
        False, 
        description="Flag indicating if PDF document structure is included"
    )
    
    # Structured data for different output formats
    table_data: List[TableData] = Field(
        default_factory=list,
        description="List of table data structures if has_table is True"
    )
    
    graph_data: List[GraphData] = Field(
        default_factory=list, 
        description="List of graph/chart data structures if has_graph is True"
    )
    
    code_data: List[CodeData] = Field(
        default_factory=list,
        description="List of code snippet structures if has_code is True"
    )
    
    image_data: List[ImageData] = Field(
        default_factory=list,
        description="List of image generation requests if has_image is True"
    )
    
    pdf_document_data: Optional[PDFDocumentData] = Field(
        None,
        description="PDF document structure if has_pdf is True"
    )
    
    # Use case specific data
    use_case_data: Optional[UseCaseOutputDataUnion] = Field(
        None,
        description="Use case specific structured data that can be used for further processing"
    )
    
    def __str__(self):
        """String representation for easier debugging"""
        output = f"MODEL RESPONSE:\n{self.model_response}\n\n"
        
        if self.has_table:
            output += f"INCLUDES {len(self.table_data)} TABLE(S)\n"
        
        if self.has_graph:
            output += f"INCLUDES {len(self.graph_data)} GRAPH(S)\n"
            
        if self.has_code:
            output += f"INCLUDES {len(self.code_data)} CODE SNIPPET(S)\n"
            
        if self.has_image:
            output += f"INCLUDES {len(self.image_data)} IMAGE REQUEST(S)\n"
            
        if self.has_pdf:
            output += "INCLUDES PDF DOCUMENT STRUCTURE\n"
            
        if self.error_message:
            output += f"ERROR: {self.error_message}\n"
            
        return output




   
    
    