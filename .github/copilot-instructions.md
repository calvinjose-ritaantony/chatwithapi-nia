# Copilot Instructions for chatwithapi-nia

## Project Overview

This project is a Generative AI powered application built with FastAPI that integrates with Azure OpenAI and other AI services. The application provides a web interface for users to interact with custom GPT models, supports WebSockets for real-time communication, and includes authentication via Microsoft Identity.

## Architecture

### Core Components

1. **Backend (FastAPI)**:
   - `main.py`: Entry point with app configuration, middleware setup, and router integration
   - `routes/`: Directory containing API endpoints
     - `gpt_routes_secured.py`: REST endpoints with authentication
     - `gpt_routes_secured_websockets.py`: WebSocket endpoints with authentication
     - `gpt_routes_unsecured.py`: Public endpoints
     - `ilama32_routes.py`: Integration with Llama models

2. **Frontend**:
   - `static/js/script.js`: REST-based frontend logic
   - `static/js/script_ws.js`: WebSocket-based frontend logic
   - Templates in `templates/` directory using Jinja2

3. **Services**:
   - `azure_openai_utils.py`: Azure OpenAI integration
   - `mongo_service.py`: MongoDB data access
   - `auth_msal.py`: Microsoft Authentication Library integration

## Key Patterns

### WebSocket vs. REST Communication

- WebSocket endpoints in `gpt_routes_secured_websockets.py` use the `/ws/` prefix
- The frontend maintains WebSocket connections via the `ConnectionManager` class
- Chat endpoints (`/chat`) specifically should use WebSockets
- Other endpoints (GPT management, etc.) use standard REST calls

### Authentication Flow

1. Microsoft Identity authentication via MSAL
2. JWT token verification in `auth_msal.py`
3. Session management via `SessionMiddleware`
4. User identification extracted from token (`user.name`)

### Data Models

- Data models in `data/` directory define schemas for MongoDB storage
- `GPTData.py`: Core model for custom GPT configurations
- `InputPrompt.py`: User input structure

## Development Workflows

### Setting Up Environment

1. Create `.env` file with required environment variables:
   ```
   SESSION_SECRET_KEY=...
   TENANT_ID=...
   CLIENT_ID=...
   CLIENT_SECRET_VALUE=...
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Local development:
   ```
   python main_local.py
   ```

2. Production:
   ```
   python main.py
   ```

## Common Patterns

### Adding New WebSocket Endpoints

When adding WebSocket endpoints:
1. Place in `gpt_routes_secured_websockets.py`
2. Use pattern: `@router.websocket("/ws/your-endpoint/{param}")`
3. Handle connections via `ConnectionManager`
4. For chat-related endpoints, use both regular and streaming variants

### Adding New REST Endpoints

For standard REST endpoints:
1. Place in `gpt_routes_secured.py`
2. Use dependency for authentication: `user: Annotated[dict, Depends(azure_scheme)]`
3. Extract username with: `loggedUser = user.name`

## Best Practices

1. Always validate user permissions before database operations
2. Use proper error handling and logging as seen in existing endpoints
3. For frontend changes, maintain the dual implementation (script.js and script_ws.js)
4. When working with chat features, ensure proper WebSocket handling for real-time responses

## Key Files for Common Tasks

- Authentication changes: `auth_msal.py`, `auth_config.py`
- Database schema changes: models in `data/` directory
- Frontend UI changes: templates in `templates/` and CSS in `static/css/`
- WebSocket logic: `ConnectionManager.py` and `gpt_routes_secured_websockets.py`
