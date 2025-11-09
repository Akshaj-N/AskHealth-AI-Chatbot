# AskHealth-AI-Chatbot

A conversational AI platform that transforms complex healthcare data queries into actionable insights. AskHealth AI bridges the gap between medical databases and natural language, enabling users to explore healthcare information through intuitive conversations backed by intelligent data analysis and dynamic visualizations.

**Tech Stack:** FastAPI, React, Gemini AI, Pinecone, Snowflake, WebSocket

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

## System Architecture

The Healthcare Data Assistant uses an agentic architecture to autonomously process and respond to data queries:

![System Architecture](assests/architecture.png)

### Key Components

1. **User Interface** (React)
   - React UI for displaying query results
   - WebSocket client for real-time communication
   - Support for visualizing text responses and plots

2. **Backend Server** (FastAPI)
   - WebSocket handler for bidirectional communication
   - Application core for orchestrating the processing pipeline
   - Plots directory for storing generated visualizations

3. **Healthcare Assistant**
   - **Query Understanding**: Analyzes natural language inputs to determine intent
   - **Vector Search**: Retrieves relevant schema metadata from Pinecone
   - **SQL Generation**: Converts natural language to optimized SQL queries
   - **Database Query**: Executes queries against Snowflake database
   - **Response Generation**: Formats database results into natural language
   - **Plot Generator**: Creates visualizations based on query results

4. **Data Sources**
   - **Snowflake Database**: Primary source for healthcare data
   - **Pinecone Vector DB**: Stores schema metadata for semantic search
   - **Web Search**: Falls back to online sources when needed

### Data Flow

1. **User Query Submission** 
   - User enters natural language query through React UI
   - Query transmitted to backend via WebSocket

2. **Request Processing**
   - Backend checks rate limiting (15 API calls/minute)
   - System determines if query is conversational or data-focused
   - For data queries, main agent orchestrates processing

3. **Data Analysis Pipeline**
   - Schema information retrieved from database
   - Relevant metadata fetched via vector search in Pinecone
   - SQL query generated and executed against Snowflake
   - Results formatted into natural language response
   - Visualizations created when appropriate

4. **Response Delivery**
   - Complete response package returned to frontend via WebSocket
   - User sees results in React UI with any visualization links
   - Response stored in conversation history for context

This architecture enables seamless processing of complex healthcare queries through an intelligent, single-agent system using specialized tools.

## User Interface

![User Interface](assests/frontend.png)

## Installation Guide

### System Requirements

- Python 3.9 or higher
- Node.js 16 or higher
- Active Snowflake account 
- Optional: Docker and Docker Compose

### Initial Setup

**1. Get the code**
```bash
git clone https://github.com/Akshaj-N/askhealth-ai-chatbot.git
cd askhealth-ai-chatbot
```

**2. Environment configuration**
```bash
# Create environment files from templates
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
```

**3. Configure credentials**

Edit your `.env` files with the following:

```bash
# Snowflake connection
SNOWFLAKE_USER=username
SNOWFLAKE_PASSWORD=secure_password
SNOWFLAKE_ACCOUNT=account_identifier
SNOWFLAKE_DATABASE=healthcare_db
SNOWFLAKE_SCHEMA=public
SNOWFLAKE_WAREHOUSE=compute_wh
SNOWFLAKE_ROLE=analyst_role

# API credentials
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=healthcare_schema
GEMINI_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key

# Application settings
REACT_APP_API_URL=http://localhost:8000
```

### Backend Configuration

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate  # Unix/macOS
# OR
venv\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt

# Launch server
python app.py
```

Backend runs at: `http://localhost:8000`

### Frontend Configuration

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

Frontend runs at: `http://localhost:3000`

### Docker Alternative

Run the complete stack with a single command:

```bash
docker-compose up -d
```




