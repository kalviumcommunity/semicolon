# 🛠️ Semicolon – AI-Powered Project Setup Automation

> **Repository**: [kalviumcommunity/semicolon](https://github.com/kalviumcommunity/semicolon)

Semicolon is a CLI-native AI agent that automates modern development project setup by intelligently parsing official documentation and executing verified commands. Built with **Gemini API** and **LangChain**, it eliminates the tedious process of manual project scaffolding.

```bash
semicolon init https://nextjs.org/docs/getting-started
# ➡️ Automated Next.js project setup in minutes
```

---

## 🎯 Project Overview

### Problem Statement
Developers waste **8+ hours weekly** on repetitive project setup tasks:
- Re-reading constantly evolving documentation
- Fighting with outdated boilerplate templates
- Debugging version conflicts and dependency issues
- Manual copy-pasting of setup commands

### Solution
Semicolon automates the entire workflow through:
- **Intelligent Documentation Parsing** (RAG)
- **Structured Command Generation** (Gemini API)
- **Safe Execution with Verification** (LangChain)
- **Interactive CLI Interface**

---

## 🧠 Core Implementation Concepts

### 1. 💬 Prompting Strategy

#### System Prompt Design
Semicolon uses a carefully engineered system prompt with Gemini API to ensure consistent, safe behavior:

```python
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["framework", "documentation_context"],
    template="""
    You are Semicolon, an expert CLI assistant for project scaffolding.
    
    CORE PRINCIPLES:
    - Generate only safe, verified commands from official documentation
    - Respond in structured JSON format with reasoning
    - Prioritize modern best practices and latest versions
    - Never hallucinate commands or configurations
    
    CONTEXT:
    Framework: {framework}
    Documentation: {documentation_context}
    
    CONSTRAINTS:
    - Execute only non-destructive operations
    - Verify each step before proceeding
    - Provide clear reasoning for every action
    - Ask for clarification when uncertain
    """
)
```

#### User Prompt Processing
User inputs are transformed into structured prompts with context injection:

```python
class PromptProcessor:
    def process_user_input(self, user_command: str, docs_context: str):
        return {
            "task": self.extract_task(user_command),
            "framework": self.identify_framework(user_command),
            "context": docs_context,
            "environment": self.detect_environment(),
            "preferences": self.load_user_preferences()
        }
```

**Example Transformation:**
- **Input**: `semicolon init https://vitejs.dev/guide/`
- **Processed**: Structured prompt with Vite documentation context, environment detection, and task specification

### 2. 📊 Structured Output System

Semicolon implements a **JSON-based reasoning loop** using Gemini's structured output capabilities:

```python
from pydantic import BaseModel
from typing import Literal, Optional

class SemicolonResponse(BaseModel):
    mode: Literal["THINK", "ACTION", "CREATE_FILE", "VERIFY", "CLARIFY", "OUTPUT"]
    content: str
    metadata: Optional[dict] = None
    reasoning: str
    safety_level: Literal["safe", "review", "dangerous"]
    next_steps: Optional[list] = None

# Gemini API configuration for structured output
gemini_config = {
    "response_schema": SemicolonResponse.model_json_schema(),
    "temperature": 0.1,  # Low for consistency
    "top_p": 0.9,       # Focused generation
    "max_output_tokens": 2048
}
```

#### Response Flow Examples:

**🧠 THINK Mode** - Planning Phase
```json
{
  "mode": "THINK",
  "content": "Setting up Vite project requires: 1) npm create vite, 2) dependency installation, 3) configuration",
  "reasoning": "Following official Vite documentation workflow",
  "safety_level": "safe",
  "next_steps": ["execute_create_command", "install_dependencies"]
}
```

**⚡ ACTION Mode** - Command Execution
```json
{
  "mode": "ACTION",
  "content": "npm create vite@latest my-project -- --template react-ts",
  "metadata": {
    "command_type": "project_creation",
    "working_directory": "./",
    "expected_output": "project_directory"
  },
  "reasoning": "Using latest Vite with TypeScript React template",
  "safety_level": "safe"
}
```

### 3. 🔧 Function Calling Implementation

Semicolon leverages LangChain's function calling capabilities with Gemini API:

```python
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI

class SemicolonTools:
    @staticmethod
    def execute_command(command: str, working_dir: str = "./") -> dict:
        """Execute shell commands with safety validation"""
        safety_check = CommandValidator.validate(command)
        if not safety_check.is_safe:
            return {"error": "Command failed safety validation", "details": safety_check.issues}
        
        result = subprocess.run(command, shell=True, cwd=working_dir, 
                              capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
    
    @staticmethod
    def create_file(path: str, content: str) -> dict:
        """Create files with content validation"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            return {"success": True, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def verify_setup(project_path: str, expected_files: list) -> dict:
        """Verify project setup completion"""
        missing_files = []
        for file in expected_files:
            if not os.path.exists(os.path.join(project_path, file)):
                missing_files.append(file)
        
        return {
            "complete": len(missing_files) == 0,
            "missing_files": missing_files,
            "verification_time": datetime.now().isoformat()
        }

# Initialize LangChain agent with Gemini
tools = [
    Tool(name="execute_command", func=SemicolonTools.execute_command, 
         description="Execute shell commands safely"),
    Tool(name="create_file", func=SemicolonTools.create_file,
         description="Create files with specified content"),
    Tool(name="verify_setup", func=SemicolonTools.verify_setup,
         description="Verify project setup completion")
]

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
agent = initialize_agent(tools, llm, agent_type="structured-chat-zero-shot-react-description")
```

### 4. 📚 RAG (Retrieval-Augmented Generation)

Semicolon implements a sophisticated RAG pipeline for real-time documentation processing:

```python
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

class DocumentationRAG:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "```", "##", "###"]
        )
        self.vectorstore = None
    
    async def process_documentation(self, url: str) -> dict:
        """Process documentation URL and create searchable knowledge base"""
        # 1. Load and parse documentation
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        # 2. Extract setup-specific content
        setup_docs = self.filter_setup_content(documents)
        
        # 3. Split into chunks
        chunks = self.text_splitter.split_documents(setup_docs)
        
        # 4. Create vector embeddings
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./docs_db"
        )
        
        return {
            "processed_chunks": len(chunks),
            "knowledge_base_ready": True,
            "source_url": url
        }
    
    def retrieve_setup_instructions(self, query: str, k: int = 5) -> list:
        """Retrieve relevant setup instructions using semantic search"""
        if not self.vectorstore:
            raise ValueError("Documentation not processed yet")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.get_relevant_documents(query)
        
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": self.calculate_relevance(doc, query)
            }
            for doc in relevant_docs
        ]
    
    def filter_setup_content(self, documents) -> list:
        """Filter documents to focus on setup/installation content"""
        setup_keywords = [
            "installation", "getting started", "setup", "create", 
            "init", "scaffold", "dependencies", "requirements"
        ]
        
        filtered_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            if any(keyword in content_lower for keyword in setup_keywords):
                filtered_docs.append(doc)
        
        return filtered_docs
```

#### RAG Workflow Integration:

```python
class SemicolonAgent:
    def __init__(self):
        self.rag = DocumentationRAG()
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    async def process_init_command(self, docs_url: str, project_name: str):
        # 1. Process documentation with RAG
        await self.rag.process_documentation(docs_url)
        
        # 2. Retrieve relevant setup instructions
        setup_context = self.rag.retrieve_setup_instructions(
            f"create new project {project_name} installation setup"
        )
        
        # 3. Generate structured response with context
        response = await self.llm.agenerate([
            self.build_prompt_with_context(setup_context, project_name)
        ])
        
        return response
```

---

## 📊 Evaluation Criteria Implementation

### ✅ Correctness
- **Command Validation**: Multi-layer safety checks before execution
- **Documentation Verification**: Cross-reference with official sources
- **Output Verification**: Automated testing of generated project structure
- **Version Compatibility**: Real-time dependency version checking

### ⚡ Efficiency
- **Parallel Processing**: Concurrent documentation parsing and command preparation
- **Caching Strategy**: Vector embeddings cached for repeated framework usage
- **Optimized Retrieval**: Semantic search with relevance scoring
- **Minimal API Calls**: Batch operations and intelligent prompt engineering

### 📈 Scalability
- **Modular Architecture**: Plugin-based framework support
- **Vector Database**: Persistent storage for documentation embeddings
- **Async Operations**: Non-blocking I/O for concurrent project setups
- **Resource Management**: Intelligent memory and API quota management

---

## 🚀 Example Implementation Workflow

```bash
$ semicolon init https://nextjs.org/docs/getting-started --name my-app
```

**Internal Process:**
1. **RAG Processing**: Parse Next.js documentation, create embeddings
2. **Context Retrieval**: Find relevant setup instructions
3. **Structured Planning**: Generate JSON response with THINK mode
4. **Function Calling**: Execute commands via LangChain tools
5. **Verification**: Confirm project structure and dependencies
6. **Output**: Provide summary and next steps

---

## 🛠️ Technical Stack

- **AI Model**: Google Gemini Pro API
- **Framework**: LangChain for agent orchestration
- **Vector Database**: Chroma for document embeddings
- **CLI Interface**: Python Click framework
- **Safety Layer**: Custom command validation system
- **Testing**: Pytest with mock framework integrations

---

## 📁 Project Architecture

```
semicolon/
├── src/
│   ├── core/
│   │   ├── agent.py          # Main Gemini-powered agent
│   │   ├── prompts.py        # System & user prompt templates
│   │   └── tools.py          # LangChain function definitions
│   ├── rag/
│   │   ├── processor.py      # Documentation RAG pipeline
│   │   ├── embeddings.py     # Vector embedding management
│   │   └── retriever.py      # Context retrieval system
│   ├── cli/
│   │   ├── commands.py       # CLI command handlers
│   │   └── interface.py      # User interaction layer
│   └── utils/
│       ├── safety.py         # Command safety validation
│       └── verification.py   # Setup verification tools
├── tests/
├── docs/
└── requirements.txt
```

---

*Built with ❤️ using Gemini API and LangChain - Automating development workflows for the modern era.*
