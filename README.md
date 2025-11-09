# AskHealth-AI-Chatbot

A conversational AI platform that transforms complex healthcare data queries into actionable insights. AskHealth AI bridges the gap between medical databases and natural language, enabling users to explore healthcare information through intuitive conversations backed by intelligent data analysis and dynamic visualizations.

**Tech Stack:** 
- FastAPI
- React
- Gemini AI
- Pinecone
- Snowflake
- WebSocket

## Key Features

- **Conversational Data Queries**: Transform questions like "What's the average surgery cost?" into precise database operations
- **Intelligent SQL Generation**: Automatically converts natural language into optimized database queries
- **Dynamic Visualizations**: Generates charts and plots from query outputs
- **Semantic Search**: Uses vector embeddings to understand database schema and relationships
- **Hybrid Knowledge Base**: Combines structured database queries with web search for comprehensive answers
- **Real-Time Interactions**: WebSocket-powered instant communication between frontend and backend
- **Context-Aware Conversations**: Maintains dialogue history for follow-up questions
- **Professional Interface**: Clean, responsive UI designed for healthcare professionals

## Project Structure

```
AskHealth-AI-Chatbot/
│
├── backend/                           # Server-side application
│   ├── app.py                         # FastAPI entry point
│   ├── final_agentic_memory_cleaned.py  # AI agent orchestration
│   ├── gen_color_plot.py              # Chart generation engine
│   ├── snowflake_connector.py         # Database interface
│   └── web_search_handler.py          # External search integration
│
├── frontend/                          # Client-side application
│   ├── public/                        # Static resources
│   └── src/
│       ├── components/                # Reusable UI elements
│       ├── contexts/                  # State management
│       ├── services/                  # Backend communication
│       └── styles/                    # Design system
│
├── vector/                            # Vector store utilities
│   └── database/                      # Embedding management
│
└── docker-compose.yml                 # Container orchestration
```







