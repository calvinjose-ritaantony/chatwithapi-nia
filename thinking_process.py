REASONING_DATA = {
      "SEARCHING_ORDERS": [
        {
          "step": "Request Validation",
          "action": "Validating order search request parameters",
          "details": "Checking for order ID, username, or date parameters"
        },
        {
          "step": "Query Formation",
          "action": "Forming search query for order database",
          "details": "Building semantic search query for order-dataset-index"
        },
        {
          "step": "Context Retrieval",
          "action": "Retrieving order information",
          "details": "Fetching order details from Azure Cognitive Search"
        },
        {
          "step": "Response Processing",
          "action": "Processing order information",
          "details": "Formatting order details for user-friendly display"
        },
        {
          "step": "Follow-up Generation",
          "action": "Generating relevant follow-up questions",
          "details": "Creating context-aware follow-up questions based on order status"
        }
      ],"TRACK_ORDERS_TKE": [
        {
          "step": "Request Validation",
          "action": "Validating TKE order tracking parameters",
          "details": "Checking for solution ID or reference ID"
        },
        {
          "step": "Data Integration",
          "action": "Connecting to TKE systems",
          "details": "Establishing connection with nia-tke-rag-index"
        },
        {
          "step": "Context Aggregation",
          "action": "Gathering order context",
          "details": "Collecting information from multiple TKE data points"
        },
        {
          "step": "Status Analysis",
          "action": "Analyzing order status",
          "details": "Evaluating current state in order lifecycle"
        },
        {
          "step": "Response Compilation",
          "action": "Compiling tracking information",
          "details": "Formatting tracking details with technical specifications"
        }
      ],
      "SUMMARIZE_PRODUCT_REVIEWS": [
        {
          "step": "Review Collection",
          "action": "Gathering product reviews",
          "details": "Fetching reviews from order-dataset-index for specified product"
        },
        {
          "step": "Sentiment Analysis",
          "action": "Analyzing review sentiments",
          "details": "Processing customer feedback for positive/negative patterns"
        },
        {
          "step": "Feature Extraction",
          "action": "Identifying key product features",
          "details": "Extracting commonly mentioned product attributes"
        },
        {
          "step": "Statistical Compilation",
          "action": "Computing review metrics",
          "details": "Calculating average ratings and feature satisfaction scores"
        },
        {
          "step": "Summary Generation",
          "action": "Creating review summary",
          "details": "Formatting comprehensive review analysis with key insights"
        }
      ],"COMPLAINTS_AND_FEEDBACK": [
        {
          "step": "Complaint Validation",
          "action": "Validating complaint details",
          "details": "Checking complaint type and priority from nia-complaints-and-feedback-index"
        },
        {
          "step": "Historical Analysis",
          "action": "Checking complaint history",
          "details": "Reviewing previous similar complaints and resolutions"
        },
        {
          "step": "Category Classification",
          "action": "Classifying complaint type",
          "details": "Categorizing complaint for appropriate handling"
        },
        {
          "step": "Solution Matching",
          "action": "Finding resolution patterns",
          "details": "Matching complaint with standard resolution templates"
        },
        {
          "step": "Response Formulation",
          "action": "Preparing response",
          "details": "Generating personalized response with resolution steps"
        }
      ],"REVIEW_BYTES": [
        {
          "step": "Review Data Fetch",
          "action": "Accessing watch reviews",
          "details": "Retrieving reviews from nia-review-bytes-index"
        },
        {
          "step": "Brand Analysis",
          "action": "Processing brand-specific data",
          "details": "Analyzing brand reputation and common feedback"
        },
        {
          "step": "Technical Evaluation",
          "action": "Assessing technical aspects",
          "details": "Analyzing comments about watch specifications and features"
        },
        {
          "step": "Price-Value Analysis",
          "action": "Evaluating price perspectives",
          "details": "Analyzing value-for-money feedback patterns"
        },
        {
          "step": "Comparison Generation",
          "action": "Creating comparative analysis",
          "details": "Generating structured comparison of watch features and ratings"
        }
      ],"MANAGE_TICKETS": [
        {
          "step": "Ticket Validation",
          "action": "Validating ticket details",
          "details": "Checking ticket ID and status in nia-tke-incidents-index"
        },
        {
          "step": "Priority Assessment",
          "action": "Evaluating ticket priority",
          "details": "Determining urgency based on incident parameters"
        },
        {
          "step": "Chat Analysis",
          "action": "Processing conversation logs",
          "details": "Analyzing customer-agent interactions for context"
        },
        {
          "step": "Solution Identification",
          "action": "Finding resolution path",
          "details": "Determining appropriate action based on incident type"
        },
        {
          "step": "Status Update",
          "action": "Updating ticket status",
          "details": "Preparing status report with next steps"
        }
      ],"DOC_SEARCH": [
        {
          "step": "Query Analysis",
          "action": "Processing search query",
          "details": "Analyzing search terms for nia-pdf-index"
        },
        {
          "step": "Document Scanning",
          "action": "Searching document repository",
          "details": "Performing semantic search across documentation"
        },
        {
          "step": "Relevance Ranking",
          "action": "Ranking search results",
          "details": "Ordering results by relevance to query"
        },
        {
          "step": "Context Extraction",
          "action": "Extracting relevant passages",
          "details": "Identifying most relevant document sections"
        },
        {
          "step": "Response Formatting",
          "action": "Formatting search results",
          "details": "Preparing structured response with document references"
        }
      ],"CUSTOMIZED_RECOMMENDATIONS": [
        {
          "step": "User Profile Analysis",
          "action": "Analyzing purchase history",
          "details": "Retrieving customer's past orders from order-dataset-index"
        },
        {
          "step": "Pattern Recognition",
          "action": "Identifying buying patterns",
          "details": "Analyzing frequency and categories of purchases"
        },
        {
          "step": "Similarity Matching",
          "action": "Finding similar products",
          "details": "Matching user preferences with product database"
        },
        {
          "step": "Price Range Analysis",
          "action": "Evaluating price preferences",
          "details": "Determining optimal price range for recommendations"
        },
        {
          "step": "Recommendation Generation",
          "action": "Creating personalized suggestions",
          "details": "Compiling list of relevant product recommendations"
        }
      ],"GENERATE_REPORTS": [
        {
          "step": "Data Collection",
          "action": "Gathering report data",
          "details": "Fetching sales and inventory data from order-dataset-index"
        },
        {
          "step": "Metric Calculation",
          "action": "Computing key metrics",
          "details": "Calculating sales trends and performance indicators"
        },
        {
          "step": "Time Period Analysis",
          "action": "Analyzing temporal patterns",
          "details": "Identifying trends across specified time periods"
        },
        {
          "step": "Comparison Generation",
          "action": "Creating comparative analysis",
          "details": "Comparing current metrics with historical data"
        },
        {
          "step": "Report Formatting",
          "action": "Structuring report content",
          "details": "Organizing data into readable report format"
        }
      ],"GENERATE_MAIL_PROMOTION": [
        {
          "step": "Audience Segmentation",
          "action": "Identifying target audience",
          "details": "Analyzing customer segments from order-dataset-index"
        },
        {
          "step": "Offer Analysis",
          "action": "Determining promotion details",
          "details": "Evaluating promotional offers and discounts"
        },
        {
          "step": "Content Personalization",
          "action": "Customizing email content",
          "details": "Tailoring message based on customer preferences"
        },
        {
          "step": "Template Selection",
          "action": "Choosing email template",
          "details": "Selecting appropriate promotional template"
        },
        {
          "step": "Mail Generation",
          "action": "Creating final email",
          "details": "Compiling personalized promotional content"
        }
      ],"GENERATE_MAIL_ORDERS": [
        {
          "step": "Order Verification",
          "action": "Validating order details",
          "details": "Checking order information in nia-generate-mail-index"
        },
        {
          "step": "Mail Type Detection",
          "action": "Determining email type",
          "details": "Identifying whether confirmation, shipping, or thank you mail"
        },
        {
          "step": "Customer Data Retrieval",
          "action": "Fetching customer information",
          "details": "Getting recipient details and preferences"
        },
        {
          "step": "Content Assembly",
          "action": "Compiling email content",
          "details": "Gathering order-specific details and tracking information"
        },
        {
          "step": "Template Application",
          "action": "Formatting email",
          "details": "Applying appropriate email template with order details"
        }
      ],"PRODUCT_INFORMATION": [
        {
          "step": "Product Identification",
          "action": "Locating product details",
          "details": "Finding product in order-dataset-index"
        },
        {
          "step": "Specification Retrieval",
          "action": "Gathering technical details",
          "details": "Collecting product specifications and features"
        },
        {
          "step": "Availability Check",
          "action": "Checking stock status",
          "details": "Verifying current product availability"
        },
        {
          "step": "Price Verification",
          "action": "Confirming pricing",
          "details": "Retrieving current price and any applicable discounts"
        },
        {
          "step": "Information Compilation",
          "action": "Formatting product details",
          "details": "Organizing product information for display"
        }
      ],"SEASONAL_SALES": [
        {
          "step": "Season Identification",
          "action": "Determining sale period",
          "details": "Identifying seasonal campaign from nia-seasonal-sales-index"
        },
        {
          "step": "Sales Data Analysis",
          "action": "Processing seasonal metrics",
          "details": "Analyzing performance during specific season"
        },
        {
          "step": "Trend Detection",
          "action": "Identifying seasonal patterns",
          "details": "Analyzing year-over-year seasonal trends"
        },
        {
          "step": "Inventory Impact",
          "action": "Assessing stock levels",
          "details": "Evaluating inventory movement during season"
        },
        {
          "step": "Performance Summary",
          "action": "Generating seasonal report",
          "details": "Compiling comprehensive seasonal analysis"
        }
      ],"HANDLE_FAQS": [
        {
          "step": "Question Analysis",
          "action": "Processing FAQ query",
          "details": "Analyzing question from nia-faq-index"
        },
        {
          "step": "Category Matching",
          "action": "Identifying FAQ category",
          "details": "Matching question to relevant FAQ section"
        },
        {
          "step": "Answer Retrieval",
          "action": "Finding relevant response",
          "details": "Searching for most appropriate answer"
        },
        {
          "step": "Context Enhancement",
          "action": "Adding related information",
          "details": "Including additional relevant details"
        },
        {
          "step": "Response Formation",
          "action": "Formatting FAQ response",
          "details": "Preparing clear and concise answer"
        }
      ],"INTELLIGENT_INCIDENT_RESOLUTION": [
        {
          "step": "Incident Classification",
          "action": "Categorizing incident type",
          "details": "Determining nature of reported issue"
        },
        {
          "step": "Priority Assignment",
          "action": "Setting incident priority",
          "details": "Evaluating urgency and impact"
        },
        {
          "step": "Solution Search",
          "action": "Finding resolution steps",
          "details": "Searching knowledge base for similar incidents"
        },
        {
          "step": "Action Plan Creation",
          "action": "Developing resolution plan",
          "details": "Creating step-by-step resolution guide"
        },
        {
          "step": "Response Preparation",
          "action": "Formatting solution",
          "details": "Preparing detailed resolution instructions"
        }
      ],"ANALYZE_SPENDING_PATTERNS": [
        {
          "step": "Data Extraction",
          "action": "Gathering transaction data",
          "details": "Retrieving purchase history from order-dataset-index"
        },
        {
          "step": "Pattern Recognition",
          "action": "Analyzing spending behavior",
          "details": "Identifying recurring purchase patterns"
        },
        {
          "step": "Category Analysis",
          "action": "Evaluating product categories",
          "details": "Analyzing distribution across categories"
        },
        {
          "step": "Temporal Analysis",
          "action": "Checking time-based patterns",
          "details": "Analyzing purchase frequency and timing"
        },
        {
          "step": "Insight Generation",
          "action": "Creating spending summary",
          "details": "Compiling comprehensive spending analysis"
        }
      ],"PRODUCT_COMPARISON": [
        {
          "step": "Product Selection",
          "action": "Identifying products",
          "details": "Locating products for comparison in order-dataset-index"
        },
        {
          "step": "Feature Extraction",
          "action": "Gathering specifications",
          "details": "Collecting detailed product features"
        },
        {
          "step": "Price Analysis",
          "action": "Comparing prices",
          "details": "Analyzing price points and value proposition"
        },
        {
          "step": "Difference Detection",
          "action": "Identifying distinctions",
          "details": "Highlighting key differences between products"
        },
        {
          "step": "Comparison Creation",
          "action": "Formatting comparison",
          "details": "Creating structured comparison table"
        }
      ],"TRACK_ORDERS": [
        {
          "step": "Order Identification",
          "action": "Locating order details",
          "details": "Finding order in order-dataset-index"
        },
        {
          "step": "Status Check",
          "action": "Retrieving current status",
          "details": "Checking latest order status"
        },
        {
          "step": "Timeline Analysis",
          "action": "Processing order timeline",
          "details": "Analyzing order progress stages"
        },
        {
          "step": "Delivery Estimation",
          "action": "Calculating delivery time",
          "details": "Estimating arrival based on current status"
        },
        {
          "step": "Update Compilation",
          "action": "Preparing status report",
          "details": "Formatting tracking information"
        }
      ],"CREATE_PRODUCT_DESCRIPTION": [
        {
          "step": "Product Analysis",
          "action": "Gathering product information",
          "details": "Collecting key features and specifications from order-dataset-index"
        },
        {
          "step": "USP Identification",
          "action": "Identifying unique features",
          "details": "Determining key selling points"
        },
        {
          "step": "Target Audience Analysis",
          "action": "Analyzing customer segment",
          "details": "Determining appropriate tone and language"
        },
        {
          "step": "SEO Optimization",
          "action": "Incorporating keywords",
          "details": "Adding relevant search terms"
        },
        {
          "step": "Description Generation",
          "action": "Creating final content",
          "details": "Composing engaging product description"
        }
      ],"CUSTOMER_COMPLAINTS": [
        {
          "step": "Complaint Reception",
          "action": "Processing complaint details",
          "details": "Recording complaint from order-dataset-index"
        },
        {
          "step": "Severity Assessment",
          "action": "Evaluating urgency",
          "details": "Determining complaint priority level"
        },
        {
          "step": "History Check",
          "action": "Reviewing customer history",
          "details": "Checking previous interactions and complaints"
        },
        {
          "step": "Solution Identification",
          "action": "Finding resolution path",
          "details": "Determining appropriate resolution steps"
        },
        {
          "step": "Response Preparation",
          "action": "Crafting response",
          "details": "Creating empathetic and solution-focused reply"
        }
      ],"GENTELL_WOUND_ADVISOR":[
        {
          "step": "Patient Profile Analysis",
          "action": "Understanding patient background",
          "details": "Collecting details such as age, comorbidities (e.g., diabetes, vascular disease), and overall health status"
        },
        {
          "step": "Wound Assessment",
          "action": "Clarifying wound type and severity",
          "details": "Asking targeted questions about wound type (pressure ulcer, diabetic foot ulcer, venous ulcer, skin tear), size, depth, and healing stage"
        },
        {
          "step": "Exudate & Infection Check",
          "action": "Evaluating wound drainage and infection signs",
          "details": "Determining exudate level (low, moderate, heavy) and checking for odor, redness, or other infection indicators"
        },
        {
          "step": "Product Category Matching",
          "action": "Identifying suitable Gentell product types",
          "details": "Mapping wound characteristics to product categories (foam dressings, alginate dressings, hydrocolloids, collagen, etc.)"
        },
        {
          "step": "Safety & Contraindications",
          "action": "Ensuring product suitability",
          "details": "Cross-checking against contraindications such as allergies, wound depth, or patient conditions"
        },
        {
          "step": "Recommendation Generation",
          "action": "Providing wound care guidance",
          "details": "Compiling recommended Gentell products with usage instructions, clinical rationale, and wound care tips"
        }
]
}