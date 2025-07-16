import json
from typing import List, Dict, Any, Union, Optional
from standalone_programs.ModelResponse import ModelResponse, TableData, GraphData, ImageData, PDFDocumentData, CodeData, UseCaseOutputDataUnion

class ModelResponseUtils:
    """Utility class for working with ModelResponse structures"""
    
    @staticmethod
    def create_table_from_data(data: List[Dict[str, Any]], caption: Optional[str] = None) -> TableData:
        """
        Create a TableData structure from a list of dictionaries
        
        Args:
            data: List of dictionaries where each dict represents a row
            caption: Optional caption for the table
            
        Returns:
            TableData object
        """
        if not data:
            return TableData(headers=[], rows=[], caption=caption, use_case_type="GENERIC")
        
        # Extract headers from the first data item
        headers = list(data[0].keys())
        
        # Convert each data item to a row
        rows = []
        for item in data:
            row = [item.get(header, "") for header in headers]
            rows.append(row)
            
        return TableData(
            headers=headers,
            rows=rows,
            caption=caption,
            use_case_type="GENERIC"
        )
    
    @staticmethod
    def create_bar_chart(title: str, labels: List[str], data: List[float], 
                          x_label: Optional[str] = None, y_label: Optional[str] = None,
                          description: Optional[str] = None) -> GraphData:
        """
        Create a bar chart GraphData structure
        
        Args:
            title: Chart title
            labels: Category labels
            data: Data values
            x_label: Optional X-axis label
            y_label: Optional Y-axis label
            description: Optional chart description
            
        Returns:
            GraphData object configured as a bar chart
        """
        return GraphData(
            chart_type="bar",
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            labels=labels,
            datasets=[{
                "label": title,
                "data": data,
                "backgroundColor": "rgba(75, 192, 192, 0.6)"
            }],
            description=description,
            options={
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        )
    
    @staticmethod
    def create_line_chart(title: str, labels: List[str], datasets: List[Dict[str, Any]],
                           x_label: Optional[str] = None, y_label: Optional[str] = None,
                           description: Optional[str] = None) -> GraphData:
        """
        Create a line chart GraphData structure with multiple datasets
        
        Args:
            title: Chart title
            labels: X-axis labels (usually time periods)
            datasets: List of dataset objects with label, data, and optional styling
            x_label: Optional X-axis label
            y_label: Optional Y-axis label
            description: Optional chart description
            
        Returns:
            GraphData object configured as a line chart
        """
        return GraphData(
            chart_type="line",
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            labels=labels,
            datasets=datasets,
            description=description,
            options={
                "tension": 0.3,  # Curve smoothness
                "scales": {
                    "y": {
                        "beginAtZero": True
                    }
                }
            }
        )
    
    @staticmethod
    def create_pie_chart(title: str, labels: List[str], data: List[float],
                          description: Optional[str] = None) -> GraphData:
        """
        Create a pie chart GraphData structure
        
        Args:
            title: Chart title
            labels: Category labels
            data: Data values
            description: Optional chart description
            
        Returns:
            GraphData object configured as a pie chart
        """
        # Generate some pleasing colors for pie segments
        colors = [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)'
        ]
        
        # Repeat colors if we have more data points than colors
        background_colors = []
        for i in range(len(data)):
            background_colors.append(colors[i % len(colors)])
        
        return GraphData(
            chart_type="pie",
            title=title,
            labels=labels,
            x_axis_label=None,
            y_axis_label=None,
            datasets=[{
                "label": title,
                "data": data,
                "backgroundColor": background_colors
            }],
            description=description,
            options={
                "responsive": True
            }
        )
    
    @staticmethod
    def create_image_request(prompt: str, alt_text: str, 
                              width: int = 1024, height: int = 1024,
                              style: Optional[str] = None,
                              description: Optional[str] = None) -> ImageData:
        """
        Create an image generation request
        
        Args:
            prompt: Description of the image to generate
            alt_text: Accessibility alt text
            width: Image width
            height: Image height
            style: Optional style directive
            description: Optional description of the image's purpose
            
        Returns:
            ImageData object
        """
        return ImageData(
            prompt=prompt,
            alt_text=alt_text,
            width=width,
            height=height,
            style=style,
            description=description
        )
    
    @staticmethod
    def create_pdf_document(title: str, sections: List[Dict[str, Any]],
                             subtitle: Optional[str] = None,
                             include_toc: bool = True,
                             include_header_footer: bool = True,
                             metadata: Optional[Dict[str, str]] = None) -> PDFDocumentData:
        """
        Create a PDF document structure
        
        Args:
            title: Document title
            sections: List of section objects with headings and content
            subtitle: Optional document subtitle
            include_toc: Whether to include table of contents
            include_header_footer: Whether to include headers/footers
            metadata: Optional document metadata
            
        Returns:
            PDFDocumentData object
        """
        return PDFDocumentData(
            title=title,
            subtitle=subtitle,
            sections=sections,
            include_table_of_contents=include_toc,
            include_header_footer=include_header_footer,
            metadata=metadata or {}
        )
    
    @staticmethod
    def enrich_model_response(model_response: ModelResponse, 
                               tables: Optional[List[TableData]] = None,
                               graphs: Optional[List[GraphData]] = None,
                               code_snippets: Optional[List[CodeData]] = None,
                               images: Optional[List[ImageData]] = None,
                               pdf_document: Optional[PDFDocumentData] = None) -> ModelResponse:
        """
        Enriches an existing ModelResponse with additional data structures
        
        Args:
            model_response: The base ModelResponse object
            tables: Optional list of TableData objects
            graphs: Optional list of GraphData objects
            code_snippets: Optional list of CodeData objects
            images: Optional list of ImageData objects
            pdf_document: Optional PDFDocumentData object
            
        Returns:
            The enriched ModelResponse object
        """
        if tables:
            model_response.has_table = True
            model_response.table_data.extend(tables)
            
        if graphs:
            model_response.has_graph = True
            model_response.graph_data.extend(graphs)
            
        if code_snippets:
            model_response.has_code = True
            model_response.code_data.extend(code_snippets)
            
        if images:
            model_response.has_image = True
            model_response.image_data.extend(images)
            
        if pdf_document:
            model_response.has_pdf = True
            model_response.pdf_document_data = pdf_document
            
        return model_response
        
    @staticmethod
    def get_use_case_data_for_type(use_case_type: str, data: Dict[str, Any]) -> UseCaseOutputDataUnion:
        """
        Create a UseCaseOutputDataUnion for a specific use case type
        
        Args:
            use_case_type: The use case identifier (e.g., "TRACK_ORDERS")
            data: The structured data for the use case
            
        Returns:
            UseCaseOutputDataUnion object
        """
        return UseCaseOutputDataUnion(
            use_case_type=use_case_type,
            data=data
        )
