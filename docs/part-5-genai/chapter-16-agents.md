# Chapter 16: Agentic AI and Multi-Agent Systems

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Define what constitutes an AI agent** — distinguish agents from chatbots and copilots, and articulate the four pillars of agency: perception, reasoning, action, and memory.
2. **Implement the ReAct framework** — build agents that interleave reasoning traces with tool actions, and explain why this outperforms chain-of-thought alone.
3. **Design tool-calling interfaces** — write JSON schema tool definitions, handle errors gracefully, and choose between OpenAI, Anthropic, and open-source function calling.
4. **Build stateful agent graphs with LangGraph** — design nodes, edges, conditional routing, checkpointing, and human-in-the-loop workflows.
5. **Orchestrate multi-agent systems** — implement supervisor, debate, voting, and hierarchical coordination patterns using AutoGen, CrewAI, and custom architectures.
6. **Implement agent memory systems** — distinguish working, episodic, semantic, and procedural memory and build each with appropriate data structures.
7. **Integrate tools via MCP** — understand Anthropic's Model Context Protocol, its transport layer, resource model, and why standardization matters for the agent ecosystem.
8. **Evaluate agents rigorously** — measure task success, trajectory efficiency, tool use accuracy, cost, and safety.

---

## 16.1 From Chatbots to Agents

### 16.1.1 The Evolution

The journey from simple chatbots to autonomous agents represents one of the most significant shifts in applied AI. Understanding this evolution clarifies what "agentic AI" means and why it matters.

**Stage 1: Prompt-Response (2020–2022).** Early LLM applications followed a stateless request-response pattern. A user sends a prompt; the model returns a completion. There is no memory between turns, no access to external tools, and no ability to take actions in the world. ChatGPT at launch exemplified this stage.

**Stage 2: Tool Use (2023).** The introduction of function calling by OpenAI (June 2023) marked a fundamental shift. Models could now request the execution of external functions — search the web, query a database, send an email — by outputting structured JSON describing the function call and its arguments. The application executes the function and returns the result to the model, which incorporates it into its response. This created a new loop: model reasons → model calls tool → application executes tool → result feeds back to model.

**Stage 3: Autonomous Agents (2023–present).** Agents extend tool use with autonomous multi-step planning. Rather than executing a single tool call per turn, agents can plan a sequence of actions, execute them iteratively, observe results, revise their plans, and continue until a goal is achieved — all without human intervention at each step. Projects like AutoGPT, BabyAGI, and MetaGPT demonstrated (with varying degrees of reliability) that LLMs could orchestrate complex workflows autonomously.

### 16.1.2 What Defines an Agent

An AI agent is a system that perceives its environment, reasons about its goals, takes actions to achieve those goals, and learns from the results. More formally, an agent requires four capabilities:

**Perception.** The agent must observe its environment — read user messages, parse API responses, process search results, examine file contents, interpret visual inputs.

**Reasoning.** The agent must plan its actions — decompose complex goals into sub-tasks, decide which tools to use and in what order, evaluate whether its current approach is working, and adjust its strategy when it is not.

**Action.** The agent must affect its environment — call APIs, write files, execute code, send messages, interact with other systems.

**Memory.** The agent must maintain state — remember what it has already done, store information gathered during execution, recall relevant past experiences, and track progress toward its goal.

The degree of autonomy distinguishes agents along a spectrum. A chatbot with a search tool has minimal autonomy (single tool call, human validates result). A fully autonomous agent might receive a high-level goal ("Research competitor pricing and create a comparison report"), plan a multi-step workflow, execute dozens of tool calls, handle errors and dead ends, and produce a final artifact — all without human intervention.

---

## 16.2 The ReAct Framework

### 16.2.1 Reasoning + Acting

Yao et al. (2023) introduced **ReAct** (Reasoning + Acting), a framework that interleaves chain-of-thought reasoning with tool actions. The key insight is that reasoning and acting are mutually reinforcing: reasoning helps the agent plan its next action, and action results inform the agent's next reasoning step.

The ReAct loop follows a **Thought → Action → Observation** cycle:

1. **Thought:** The agent reasons about the current state, what information it needs, and what action to take next.
2. **Action:** The agent selects and executes a tool (e.g., search, calculate, look up).
3. **Observation:** The tool returns a result, which the agent incorporates into its context.
4. **Repeat** until the agent has enough information to produce a final answer.

### 16.2.2 Formal Framework

Given:
- A set of available tools $\mathcal{T} = \{t_1, t_2, \ldots, t_n\}$
- A user query $q$
- A context buffer $C$ (initially containing $q$)

At each step $i$, the agent generates:

$$(\text{thought}_i, \text{action}_i, \text{args}_i) = \text{LLM}(C)$$

The system executes the action and appends the result:

$$\text{observation}_i = t_{\text{action}_i}(\text{args}_i)$$

$$C \leftarrow C \oplus \text{thought}_i \oplus \text{action}_i \oplus \text{observation}_i$$

The loop terminates when the agent outputs a special "Finish" action with the final answer.

### 16.2.3 Example Trace

```
Question: What is the population of the capital of the country that won the 2024
          Cricket World Cup?

Thought 1: I need to find which country won the 2024 Cricket World Cup. Let me
           search for this.
Action 1: Search["2024 Cricket World Cup winner"]
Observation 1: India won the 2024 ICC Men's T20 World Cup, defeating South Africa
               in the final.

Thought 2: India won the 2024 Cricket World Cup. The capital of India is New Delhi.
           Now I need the population of New Delhi.
Action 2: Search["population of New Delhi 2024"]
Observation 2: The population of New Delhi is approximately 32.9 million in the
               National Capital Territory.

Thought 3: I now have all the information needed to answer the question.
Action 3: Finish["The country that won the 2024 Cricket World Cup is India. Its
          capital is New Delhi, which has a population of approximately 32.9 million."]
```

### 16.2.4 Why ReAct Outperforms Chain-of-Thought

Chain-of-thought (CoT) prompting asks the model to "think step by step" but confines all reasoning to the model's parametric knowledge. When the model's knowledge is incomplete, outdated, or wrong, CoT reasoning propagates errors. ReAct breaks out of this closed loop by grounding reasoning in real-world observations via tool use.

Empirically, Yao et al. (2023) showed that ReAct outperforms CoT on knowledge-intensive tasks (HotpotQA, FEVER) by 5–10% absolute, while also providing interpretable reasoning traces that humans can audit and debug.

---

## 16.3 Tool Calling

### 16.3.1 OpenAI Function Calling

OpenAI's function calling API allows models to output structured JSON that specifies a function name and its arguments. The developer defines available tools as JSON Schema objects:

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a function
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Function: {function_name}, Args: {arguments}")
        # Execute the function and return result
```

The model may return multiple tool calls in a single response (parallel function calling), enabling efficient execution of independent operations.

### 16.3.2 Anthropic Tool Use

Anthropic's tool use follows a similar pattern but with distinct API design:

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g., 'San Francisco, CA'"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}]
)

# Process tool use blocks
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}, Input: {block.input}")
        # Execute tool and send result back
```

### 16.3.3 Open-Source Function Calling

Open-source models have rapidly developed function calling capabilities:

**Gorilla** (Patil et al., 2023) was trained specifically on API documentation and excels at generating correct API calls, including handling API version changes and deprecated parameters.

**NexusRaven** is a 13B parameter model fine-tuned specifically for function calling, achieving competitive performance with GPT-4 on tool use benchmarks.

**Hermes and Qwen function-calling models** extend open-weight models with structured output capabilities, supporting the same JSON schema tool definitions used by commercial APIs.

### 16.3.4 Error Handling

Robust tool calling requires defensive engineering:

```python
import json
from typing import Any

def execute_tool_call(tool_name: str, arguments: dict, tool_registry: dict) -> dict:
    """Execute a tool call with comprehensive error handling."""
    # Validate tool exists
    if tool_name not in tool_registry:
        return {
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(tool_registry.keys())
        }

    tool_fn = tool_registry[tool_name]

    try:
        # Execute with timeout
        result = tool_fn(**arguments)
        return {"success": True, "result": result}
    except TypeError as e:
        return {"error": f"Invalid arguments: {str(e)}"}
    except TimeoutError:
        return {"error": f"Tool '{tool_name}' timed out after 30 seconds"}
    except Exception as e:
        return {"error": f"Tool execution failed: {type(e).__name__}: {str(e)}"}
```

Key error-handling patterns include:
- **Argument validation:** Check types and required fields before execution.
- **Timeouts:** Set maximum execution times for each tool.
- **Retry with feedback:** Return errors to the model so it can correct its approach.
- **Graceful degradation:** If a tool fails, the agent should try alternative approaches.

---

## 16.4 LangChain

### 16.4.1 Architecture Overview

LangChain (Chase, 2023) is the most widely adopted framework for building LLM applications. It provides abstractions for:

**Chains** are sequences of operations — prompt construction, LLM call, output parsing — composed together. The simplest chain is `prompt | llm | parser`.

**Agents** extend chains with dynamic tool selection. Instead of a fixed sequence of operations, an agent repeatedly chooses which tool to call based on intermediate results.

**Retrievers** abstract document retrieval — from vector stores, keyword indices, or hybrid sources.

**LCEL (LangChain Expression Language)** is a declarative syntax for composing chains using the pipe (`|`) operator:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# LCEL chain composition
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple terms."
)
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

chain = prompt | model | parser

# Invoke
result = chain.invoke({"topic": "quantum computing"})

# Stream
for chunk in chain.stream({"topic": "quantum computing"}):
    print(chunk, end="", flush=True)

# Batch
results = chain.batch([
    {"topic": "quantum computing"},
    {"topic": "neural networks"},
    {"topic": "blockchain"}
])
```

### 16.4.2 When to Use and When Not To

LangChain excels for:
- Rapid prototyping of LLM applications
- Projects using multiple integrations (vector stores, tools, LLMs from different providers)
- Teams that benefit from standardized abstractions

LangChain may be overkill or counterproductive when:
- You have a simple, well-defined pipeline (direct API calls are simpler)
- You need fine-grained control over prompts and API parameters
- You are building performance-critical systems where abstraction overhead matters
- You are debugging complex interactions (deep abstraction stacks can be opaque)

The LangChain ecosystem has grown large enough that choosing the right abstraction level is itself a design decision. Many practitioners use LangChain's low-level primitives (prompt templates, output parsers) while avoiding its high-level agents in favor of LangGraph.

---

## 16.5 LangGraph

### 16.5.1 State Graphs for Agent Workflows

LangGraph (2024) reimagines agent construction as a graph problem. Instead of relying on the implicit loop of a ReAct agent, LangGraph makes the control flow explicit through a state machine:

- **State:** A typed data structure that flows through the graph, accumulating results.
- **Nodes:** Functions that read and modify the state.
- **Edges:** Connections between nodes, including conditional edges that route based on state.
- **Checkpointing:** Automatic persistence of state at each node, enabling resumption, replay, and human-in-the-loop workflows.

### 16.5.2 Core Concepts

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Define state schema
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next_step: str
    research_results: list[str]
    final_answer: str
```

The state schema defines all data that flows through the graph. The `Annotated[list, add_messages]` pattern tells LangGraph to *append* new messages rather than replace the list — a reducer pattern essential for conversation histories.

### 16.5.3 Complete Multi-Step Agent Example

Let us build a research agent that takes a question, searches for information, evaluates whether it has enough context, and produces a final answer.

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json

# State definition
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_notes: list[str]
    search_count: int
    should_continue: bool

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Node 1: Analyze query and decide action
def analyze_query(state: ResearchState) -> dict:
    """Analyze the user's question and decide on research strategy."""
    messages = state["messages"]
    research_notes = state.get("research_notes", [])

    system_msg = SystemMessage(content="""You are a research analyst.
    Analyze the user's question and the research notes gathered so far.

    If you have enough information, set should_continue to false.
    If you need more information, describe what to search for next.

    Respond in JSON: {"analysis": "...", "search_query": "...", "should_continue": true/false}
    """)

    context = f"\nResearch notes so far: {json.dumps(research_notes)}" if research_notes else ""
    response = llm.invoke([system_msg] + messages + [HumanMessage(content=context)])

    try:
        result = json.loads(response.content)
    except json.JSONDecodeError:
        result = {"analysis": response.content, "search_query": "", "should_continue": False}

    return {
        "messages": [AIMessage(content=f"Analysis: {result['analysis']}")],
        "should_continue": result.get("should_continue", False)
    }

# Node 2: Execute search
def execute_search(state: ResearchState) -> dict:
    """Execute a search based on the current analysis."""
    messages = state["messages"]
    search_count = state.get("search_count", 0)

    # In production, this would call a real search API
    last_message = messages[-1].content if messages else ""
    search_result = f"[Search result {search_count + 1} for context related to: {last_message}]"

    research_notes = state.get("research_notes", [])
    research_notes.append(search_result)

    return {
        "research_notes": research_notes,
        "search_count": search_count + 1,
        "messages": [AIMessage(content=f"Found: {search_result}")]
    }

# Node 3: Generate final answer
def generate_answer(state: ResearchState) -> dict:
    """Synthesize research notes into a final answer."""
    messages = state["messages"]
    research_notes = state.get("research_notes", [])

    system_msg = SystemMessage(content="""Synthesize the research notes into
    a comprehensive answer to the user's original question. Cite your sources.""")

    context = HumanMessage(content=f"Research notes: {json.dumps(research_notes)}")
    response = llm.invoke([system_msg] + messages + [context])

    return {"messages": [response]}

# Conditional edge: should we continue researching?
def should_continue(state: ResearchState) -> Literal["execute_search", "generate_answer"]:
    """Decide whether to continue researching or generate the final answer."""
    if state.get("should_continue", False) and state.get("search_count", 0) < 3:
        return "execute_search"
    return "generate_answer"

# Build the graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("execute_search", execute_search)
workflow.add_node("generate_answer", generate_answer)

# Add edges
workflow.add_edge(START, "analyze")
workflow.add_conditional_edges("analyze", should_continue)
workflow.add_edge("execute_search", "analyze")  # Loop back to analyze after search
workflow.add_edge("generate_answer", END)

# Compile with checkpointing
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Run the agent
config = {"configurable": {"thread_id": "research-1"}}
result = app.invoke(
    {"messages": [HumanMessage(content="What are the latest advances in protein folding?")],
     "research_notes": [], "search_count": 0, "should_continue": True},
    config=config
)
```

### 16.5.4 Human-in-the-Loop

LangGraph's checkpointing enables powerful human-in-the-loop patterns. By adding an `interrupt_before` or `interrupt_after` parameter to sensitive nodes, the graph pauses execution and waits for human approval:

```python
# Compile with interrupt before the execute_search node
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["execute_search"]  # Pause before executing searches
)

# Run until interrupt
result = app.invoke(initial_state, config)
# Graph pauses before execute_search

# Human reviews the planned search, then resumes
# (could modify state before resuming)
result = app.invoke(None, config)  # Resume from checkpoint
```

This pattern is essential for high-stakes applications — legal research, medical diagnosis assistance, financial analysis — where autonomous agent actions must be reviewed before execution.

---

## 16.6 AutoGen

### 16.6.1 Multi-Agent Conversations

AutoGen (Wu et al., 2023) from Microsoft Research introduces a framework where multiple agents communicate through natural language conversations. Instead of a single agent with tools, AutoGen creates specialized agents that collaborate through dialogue.

### 16.6.2 Architecture

AutoGen's core abstractions are:

**ConversableAgent:** The base agent class. Each agent has a name, a system message defining its role, and an LLM configuration. Agents can send messages to other agents and respond to messages they receive.

**AssistantAgent:** A ConversableAgent configured for general-purpose assistance. It can generate code, analyze data, and reason through problems.

**UserProxyAgent:** Represents the human user. It can execute code, provide human feedback, and terminate conversations.

**GroupChat:** Orchestrates a conversation among multiple agents. A GroupChatManager selects which agent speaks next based on the conversation context.

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Configure LLM
llm_config = {
    "model": "gpt-4o",
    "temperature": 0,
    "api_key": "your-api-key"
}

# Create specialized agents
researcher = AssistantAgent(
    name="Researcher",
    system_message="""You are a research specialist. Your job is to search for
    information, analyze sources, and present findings clearly. Always cite sources.""",
    llm_config=llm_config
)

analyst = AssistantAgent(
    name="Analyst",
    system_message="""You are a data analyst. Your job is to analyze data,
    identify patterns, and provide quantitative insights. Write Python code
    when analysis is needed.""",
    llm_config=llm_config
)

writer = AssistantAgent(
    name="Writer",
    system_message="""You are a technical writer. Your job is to synthesize
    research and analysis into clear, well-structured reports.""",
    llm_config=llm_config
)

# User proxy for code execution and human feedback
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # Only ask for input at termination
    code_execution_config={"work_dir": "workspace", "use_docker": False},
    max_consecutive_auto_reply=10
)

# Create group chat
group_chat = GroupChat(
    agents=[user_proxy, researcher, analyst, writer],
    messages=[],
    max_round=15,
    speaker_selection_method="auto"  # LLM selects next speaker
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# Start the conversation
user_proxy.initiate_chat(
    manager,
    message="Analyze the trends in AI research funding over the last 5 years."
)
```

### 16.6.3 Speaker Selection

AutoGen's `speaker_selection_method` determines which agent speaks next in a group chat:

- **`"auto"`:** The GroupChatManager uses the LLM to select the next speaker based on conversation context.
- **`"round_robin"`:** Agents take turns in a fixed order.
- **`"random"`:** Random selection.
- **Custom function:** A user-defined function that receives the conversation history and returns the next speaker.

The "auto" method is most flexible but adds latency (one additional LLM call per turn for speaker selection). For well-defined workflows, custom functions that encode the expected conversation flow are more efficient and reliable.

---

## 16.7 Planning Agents

### 16.7.1 Tree of Thought

Yao et al. (2023b) extended chain-of-thought prompting into a search problem with **Tree of Thought (ToT)**. Instead of generating a single reasoning chain, ToT explores multiple reasoning paths simultaneously:

1. **Decompose** the problem into intermediate thought steps.
2. **Generate** multiple candidate thoughts at each step.
3. **Evaluate** each candidate (using the LLM itself as an evaluator).
4. **Search** through the tree using BFS or DFS, pruning unpromising branches.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ThoughtNode:
    thought: str
    score: float
    children: list["ThoughtNode"]
    parent: Optional["ThoughtNode"] = None

def tree_of_thought(problem: str, llm, depth: int = 3, breadth: int = 3) -> str:
    """Implement Tree of Thought search."""
    root = ThoughtNode(thought=problem, score=1.0, children=[])

    def expand(node: ThoughtNode, current_depth: int):
        if current_depth >= depth:
            return

        # Generate multiple candidate next thoughts
        prompt = f"""Problem: {problem}
        Reasoning so far: {get_path(node)}
        Generate {breadth} different next reasoning steps.
        Return as a numbered list."""

        candidates = llm.invoke(prompt).content
        candidate_list = parse_candidates(candidates)

        for candidate in candidate_list:
            # Evaluate each candidate
            eval_prompt = f"""Rate this reasoning step (0-10):
            Problem: {problem}
            Reasoning so far: {get_path(node)}
            Next step: {candidate}
            Score:"""

            score = float(llm.invoke(eval_prompt).content.strip())
            child = ThoughtNode(
                thought=candidate, score=score,
                children=[], parent=node
            )
            node.children.append(child)

        # Prune: keep only top-k children
        node.children.sort(key=lambda x: x.score, reverse=True)
        node.children = node.children[:breadth]

        # Recursively expand best children
        for child in node.children:
            expand(child, current_depth + 1)

    expand(root, 0)
    # Return the best leaf path
    return get_best_path(root)
```

### 16.7.2 Monte Carlo Tree Search for Action Planning

MCTS, originally developed for game-playing (Silver et al., 2016), has been adapted for LLM agent planning. The four phases of MCTS apply naturally to action planning:

1. **Selection:** Starting from the root (initial state), traverse the tree by selecting actions that balance exploitation (high reward) and exploration (rarely tried), using the UCT formula:

$$\text{UCT}(a) = \frac{Q(a)}{N(a)} + c \sqrt{\frac{\ln N(\text{parent})}{N(a)}}$$

2. **Expansion:** At a leaf node, generate new candidate actions using the LLM.
3. **Simulation:** Roll out a complete plan from the new node (using the LLM to simulate future actions and outcomes).
4. **Backpropagation:** Update visit counts and reward estimates for all nodes on the path.

### 16.7.3 Plan-and-Execute Pattern

A simpler but effective planning pattern separates planning from execution:

1. **Plan:** The LLM generates a complete, numbered plan of steps to accomplish the goal.
2. **Execute:** Each step is executed sequentially, with the results fed back to the LLM.
3. **Replan:** After each step (or when a step fails), the LLM can revise the remaining plan.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def plan_and_execute(goal: str, tools: dict) -> str:
    """Plan-and-execute agent loop."""
    # Phase 1: Generate plan
    plan_prompt = f"""Create a step-by-step plan to accomplish this goal:
    Goal: {goal}

    Available tools: {list(tools.keys())}

    Return a numbered list of steps. Each step should specify which tool to use."""

    plan = llm.invoke([HumanMessage(content=plan_prompt)]).content
    steps = parse_plan(plan)

    results = []
    for i, step in enumerate(steps):
        # Phase 2: Execute each step
        exec_prompt = f"""Execute this step of the plan.

        Overall goal: {goal}
        Current step ({i+1}/{len(steps)}): {step}
        Previous results: {results}

        Specify the tool call as JSON: {{"tool": "...", "args": {{...}}}}"""

        response = llm.invoke([HumanMessage(content=exec_prompt)]).content
        tool_call = json.loads(response)
        result = tools[tool_call["tool"]](**tool_call["args"])
        results.append({"step": step, "result": result})

        # Phase 3: Check if replanning is needed
        replan_prompt = f"""Given the result of step {i+1}, should the remaining plan change?
        Result: {result}
        Remaining steps: {steps[i+1:]}

        Reply "CONTINUE" or provide a revised plan."""

        replan = llm.invoke([HumanMessage(content=replan_prompt)]).content
        if replan.strip() != "CONTINUE":
            steps = steps[:i+1] + parse_plan(replan)

    return synthesize_results(goal, results)
```

---

## 16.8 Memory Systems

### 16.8.1 Taxonomy of Agent Memory

Drawing from cognitive science, agent memory systems can be categorized into four types:

**Working Memory** is the agent's immediate context — the conversation history and any information gathered during the current task. In most implementations, this is simply the message list passed to the LLM. Working memory is limited by the model's context window.

**Episodic Memory** stores records of past interactions and experiences. When the agent encounters a similar situation, it can retrieve relevant episodes to inform its current behavior. Implementation typically uses a vector database:

```python
from datetime import datetime

class EpisodicMemory:
    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model

    def store_episode(self, task: str, actions: list, outcome: str,
                      success: bool):
        """Store a completed task as an episode."""
        episode_text = f"Task: {task}\nActions: {actions}\nOutcome: {outcome}"
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "task_type": classify_task(task)
        }
        self.vectorstore.add_texts([episode_text], metadatas=[metadata])

    def recall(self, current_task: str, k: int = 3) -> list[str]:
        """Recall relevant past episodes."""
        results = self.vectorstore.similarity_search(current_task, k=k)
        return [doc.page_content for doc in results]
```

**Semantic Memory** is the agent's long-term knowledge base — facts, concepts, and relationships that persist across tasks. This might be a RAG system, a knowledge graph, or a structured database that the agent queries as needed.

**Procedural Memory** encodes learned skills and procedures — how to accomplish specific types of tasks. This can be implemented as:
- A library of prompt templates for different task types
- Stored tool-use sequences that worked in the past
- Fine-tuned model weights that encode task-specific behavior

### 16.8.2 Memory Management

As conversations grow, working memory eventually exceeds the context window. Strategies for managing this include:

- **Sliding window:** Keep only the last $N$ messages.
- **Summarization:** Periodically summarize older messages and replace them with the summary.
- **Selective retention:** Use the LLM to identify which messages are important and discard the rest.
- **Hierarchical memory:** Store recent messages in working memory, older messages in episodic memory (vector DB), and synthesized knowledge in semantic memory (knowledge base).

```python
def manage_memory(messages: list, max_tokens: int = 8000) -> list:
    """Manage conversation memory with summarization."""
    current_tokens = count_tokens(messages)

    if current_tokens <= max_tokens:
        return messages

    # Keep system message and last N messages
    system_msg = messages[0]
    recent = messages[-10:]

    # Summarize older messages
    older = messages[1:-10]
    summary = llm.invoke([
        SystemMessage(content="Summarize this conversation concisely, "
                      "preserving key decisions and information."),
        HumanMessage(content=format_messages(older))
    ]).content

    return [system_msg,
            AIMessage(content=f"[Summary of earlier conversation: {summary}]")] + recent
```

---

## 16.9 Computer Use Agents

### 16.9.1 Browser Automation

Computer use agents interact with software systems the same way humans do — through GUIs, browsers, and file systems. This capability dramatically expands the range of tasks agents can perform.

**Playwright** provides programmatic browser control:

```python
from playwright.async_api import async_playwright
import asyncio

async def browse_and_extract(url: str, instruction: str) -> str:
    """Use Playwright to browse a webpage and extract information."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        # Take screenshot for multimodal analysis
        screenshot = await page.screenshot()

        # Extract text content
        content = await page.text_content("body")

        # Interact with elements
        # await page.click("button#submit")
        # await page.fill("input#search", "query text")

        await browser.close()
        return content
```

### 16.9.2 Code Execution

Agents that can write and execute code gain immense capability but require sandboxing for safety:

```python
import subprocess
import tempfile
import os

def execute_python_safely(code: str, timeout: int = 30) -> dict:
    """Execute Python code in a sandboxed subprocess."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {timeout} seconds"}
    finally:
        os.unlink(temp_path)
```

For production use, code execution should be containerized (Docker) or run in cloud sandboxes (E2B, Modal) to prevent agents from affecting the host system.

---

## 16.10 Multi-Agent Coordination

### 16.10.1 Coordination Patterns

**Supervisor Pattern.** A supervisor agent receives the user's request, decomposes it into sub-tasks, delegates each sub-task to a specialized worker agent, collects results, and synthesizes the final response. This is the most common pattern and is straightforward to implement with LangGraph:

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, Literal

class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    results: dict

def supervisor(state: SupervisorState) -> dict:
    """Supervisor decides which agent to call next."""
    prompt = f"""You are a supervisor managing specialized agents:
    - researcher: Searches for information
    - coder: Writes and executes code
    - writer: Produces written content

    Based on the conversation, which agent should act next?
    Or should we finish? Reply with: researcher, coder, writer, or FINISH"""

    response = llm.invoke(state["messages"] + [HumanMessage(content=prompt)])
    next_agent = response.content.strip().lower()
    return {"next_agent": next_agent}

def route_to_agent(state: SupervisorState) -> Literal[
    "researcher", "coder", "writer", "end"
]:
    next_agent = state["next_agent"]
    if next_agent == "finish":
        return "end"
    return next_agent

# Build supervisor graph
workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_node)
workflow.add_node("coder", coder_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", route_to_agent)

# All workers report back to supervisor
for agent in ["researcher", "coder", "writer"]:
    workflow.add_edge(agent, "supervisor")

workflow.add_edge("end", END)
```

**Debate/Discussion Pattern.** Multiple agents with different perspectives (or even adversarial roles) discuss a problem. Agent A proposes an answer; Agent B critiques it; Agent A revises; and so on until convergence. This pattern improves answer quality for complex reasoning tasks.

**Voting/Consensus Pattern.** Multiple agents independently attempt the same task. Their outputs are compared, and the majority answer (or the answer with highest confidence) is selected. This is analogous to ensemble methods in ML and is particularly effective for reducing hallucination.

**Hierarchical Pattern.** Agents are organized in a tree structure. A top-level agent decomposes the task and delegates to mid-level agents, which may further delegate to specialized leaf agents. This mirrors organizational structures in human teams and scales well for complex, multi-faceted tasks.

### 16.10.2 Challenges

Multi-agent systems introduce coordination challenges:
- **Communication overhead:** Each message between agents consumes tokens and adds latency.
- **Infinite loops:** Agents can get stuck in cycles of delegation without making progress.
- **Divergent goals:** Agents optimized for different objectives may produce conflicting outputs.
- **Cost explosion:** Each agent uses its own LLM calls; a multi-agent system with 5 agents and 10 rounds generates 50+ LLM calls per user query.

Mitigations include round limits, explicit termination conditions, shared state that tracks progress, and cost budgets that halt execution when exceeded.

---

## 16.11 Model Context Protocol (MCP)

### 16.11.1 The Standardization Problem

Before MCP, every LLM application built its own custom integration for each tool — a bespoke connector to a database, a custom wrapper around an API, a proprietary interface to a file system. This created an $M \times N$ integration problem: $M$ applications each needed $N$ custom tool integrations.

### 16.11.2 MCP Architecture

Anthropic's **Model Context Protocol** (Anthropic, 2024) standardizes the interface between LLM applications (clients) and tool providers (servers). The architecture follows a client-server model:

**MCP Host/Client:** The LLM application (e.g., Claude Desktop, an IDE plugin, a custom agent). The client discovers available MCP servers, connects to them, and presents their capabilities to the LLM.

**MCP Server:** A lightweight program that exposes tools, resources, and prompts through the MCP protocol. Each server provides a focused set of capabilities — a filesystem server, a database server, a search server.

### 16.11.3 Protocol Components

MCP defines three types of capabilities that servers can expose:

**Resources** are data that the server can provide — files, database records, API responses. The client can read resources to provide context to the LLM.

**Tools** are functions that the server can execute. Each tool has a name, description, and JSON Schema input definition — identical to the tool definitions used by OpenAI and Anthropic APIs.

**Prompts** are reusable prompt templates that the server provides, including parameter definitions and suggested usage.

### 16.11.4 Transport Layer

MCP supports two transport mechanisms:

- **stdio (Standard I/O):** The client spawns the server as a subprocess and communicates via stdin/stdout. Simple, low-latency, suitable for local tools.
- **SSE (Server-Sent Events) over HTTP:** The server runs as a web service. Suitable for remote tools, shared servers, and cloud deployments.

```python
# Example MCP server (simplified)
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.tool()
async def get_weather(location: str) -> list[TextContent]:
    """Get current weather for a location.

    Args:
        location: City name (e.g., 'San Francisco, CA')
    """
    # Fetch weather data from API
    weather_data = await fetch_weather_api(location)
    return [TextContent(
        type="text",
        text=f"Weather in {location}: {weather_data['temperature']}°F, "
             f"{weather_data['conditions']}"
    )]

@server.resource("weather://forecast/{location}")
async def get_forecast(location: str) -> str:
    """Get 5-day forecast for a location."""
    forecast = await fetch_forecast_api(location)
    return format_forecast(forecast)
```

### 16.11.5 Why Standardization Matters

MCP reduces the integration problem from $M \times N$ to $M + N$: each application implements the MCP client once, each tool implements the MCP server once, and they all interoperate. This creates a composable ecosystem where:
- Tool developers build once, reach all MCP-compatible applications.
- Application developers gain access to all MCP-compatible tools without custom integration.
- Users can mix and match tools across applications.

---

## 16.12 CrewAI and OpenAI Swarm

### 16.12.1 CrewAI

CrewAI provides a role-based multi-agent framework inspired by human team structures. Key abstractions:

**Agent:** Defined by a role, goal, backstory, and set of tools. The role and backstory shape the agent's behavior through system prompting.

**Task:** A specific objective assigned to an agent, with expected output format and dependencies on other tasks.

**Crew:** A team of agents working on a set of tasks, with a configurable process (sequential, hierarchical, or consensus-based).

```python
from crewai import Agent, Task, Crew, Process

# Define agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI",
    backstory="You are a seasoned researcher with a PhD in CS, known for "
              "your ability to find and synthesize complex technical information.",
    tools=[search_tool, web_scraper],
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Craft compelling, accurate technical content",
    backstory="You are an experienced technical writer who translates complex "
              "research into accessible articles.",
    tools=[],
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest developments in LLM agents. "
                "Focus on architectures, benchmarks, and real-world applications.",
    expected_output="A detailed research brief with key findings and sources.",
    agent=researcher
)

writing_task = Task(
    description="Write a comprehensive article based on the research brief.",
    expected_output="A well-structured article of approximately 2000 words.",
    agent=writer,
    context=[research_task]  # Depends on research task
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

### 16.12.2 OpenAI Swarm

OpenAI's Swarm (2024) is an experimental framework emphasizing lightweight, composable agent handoffs. Its core principles are:

**Routines:** Each agent is defined by a system prompt and a set of functions. An agent executes its routine until it completes or hands off to another agent.

**Handoffs:** An agent can transfer control to another agent by returning a special handoff object. This enables fluid transitions between specialized agents without a central coordinator.

```python
from swarm import Swarm, Agent

client = Swarm()

def transfer_to_sales():
    """Transfer the conversation to the sales agent."""
    return sales_agent

def transfer_to_support():
    """Transfer the conversation to the support agent."""
    return support_agent

triage_agent = Agent(
    name="Triage",
    instructions="You are a triage agent. Determine if the user needs sales or support.",
    functions=[transfer_to_sales, transfer_to_support]
)

sales_agent = Agent(
    name="Sales",
    instructions="You are a sales agent. Help users with pricing and purchasing.",
    functions=[get_pricing, create_order]
)

support_agent = Agent(
    name="Support",
    instructions="You are a support agent. Help users resolve technical issues.",
    functions=[search_docs, create_ticket]
)

# Run conversation
response = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "I'm having trouble with my account"}]
)
```

Swarm's design philosophy prioritizes simplicity and transparency over complex orchestration. It is best suited for customer-facing applications with clear agent boundaries.

---

## 16.13 Agent Evaluation

### 16.13.1 Evaluation Dimensions

Evaluating agents is fundamentally harder than evaluating single LLM calls because agents produce trajectories, not just outputs. Key dimensions include:

**Task Success Rate.** The most basic metric: did the agent accomplish its goal? This requires defining clear success criteria for each task. For coding agents, this might be "tests pass"; for research agents, "the answer matches the ground truth"; for customer service agents, "the issue was resolved."

**Trajectory Efficiency.** Given that the task was accomplished, how efficiently did the agent do it? Metrics include:
- Number of steps (fewer is better)
- Number of tool calls (fewer is better)
- Total tokens consumed (lower is better)
- Wall-clock time (faster is better)

An agent that solves a problem in 3 steps is better than one that takes 15 steps, even if both reach the correct answer.

**Tool Use Accuracy.** Of the tool calls the agent made, what fraction used the correct tool with correct arguments? This measures the agent's ability to select and invoke tools properly.

**Cost Per Task.** The total monetary cost of completing a task, including all LLM calls, tool executions, and infrastructure. This is critical for production viability.

**Safety Metrics.** Did the agent take any harmful actions? Did it access unauthorized resources? Did it leak sensitive information? Safety evaluation often requires red-teaming — deliberately trying to make the agent behave badly.

### 16.13.2 Evaluation Frameworks

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentEvalResult:
    task_id: str
    success: bool
    num_steps: int
    num_tool_calls: int
    total_tokens: int
    wall_time_seconds: float
    cost_usd: float
    errors: list[str]
    trajectory: list[dict]  # Full trace of thoughts, actions, observations

def evaluate_agent(agent, test_cases: list[dict]) -> list[AgentEvalResult]:
    """Evaluate an agent on a set of test cases."""
    results = []
    for case in test_cases:
        start_time = time.time()
        trajectory = []
        tokens = 0

        try:
            result = agent.run(
                task=case["task"],
                callbacks=[TrajectoryCallback(trajectory, tokens)]
            )
            success = case["eval_fn"](result, case["expected"])
        except Exception as e:
            result = None
            success = False

        eval_result = AgentEvalResult(
            task_id=case["id"],
            success=success,
            num_steps=len(trajectory),
            num_tool_calls=sum(1 for t in trajectory if t["type"] == "tool_call"),
            total_tokens=tokens,
            wall_time_seconds=time.time() - start_time,
            cost_usd=calculate_cost(tokens),
            errors=[t["error"] for t in trajectory if "error" in t],
            trajectory=trajectory
        )
        results.append(eval_result)

    return results
```

### 16.13.3 Benchmarks

- **SWE-bench** (Jimenez et al., 2024): Evaluates coding agents on real GitHub issues. The agent must read the issue, navigate the repository, and produce a code patch that resolves the issue and passes tests.
- **GAIA** (Mialon et al., 2023): General AI Assistants benchmark with questions requiring multi-step reasoning, tool use, and real-world knowledge.
- **WebArena** (Zhou et al., 2024): Evaluates web navigation agents on realistic tasks across self-hosted web applications.
- **AgentBench** (Liu et al., 2023): Comprehensive benchmark spanning operating system, database, knowledge graph, and web browsing environments.

---

## Summary

This chapter has traced the evolution from simple chatbots to autonomous agents, covering the theoretical foundations (ReAct, Tree of Thought, planning), practical frameworks (LangChain, LangGraph, AutoGen, CrewAI), infrastructure (tool calling, MCP), and evaluation. The key takeaway is that building effective agents requires not just a powerful LLM but a carefully designed system architecture: well-defined tools, robust error handling, appropriate memory systems, and rigorous evaluation.

The agent landscape is evolving rapidly. As models become more capable at reasoning, tool use, and long-horizon planning, the boundary between "copilot" and "autonomous agent" will continue to shift. The frameworks and patterns described in this chapter provide a foundation for building systems that can navigate this shifting boundary.

---

## Exercises

1. **ReAct agent.** Implement a ReAct agent from scratch (without LangChain) that can answer multi-hop questions using a search tool and a calculator tool. Test it on 10 questions from HotpotQA.

2. **Tool design.** Design a set of 5 tools for a customer support agent at an e-commerce company. Define the JSON schema for each tool, implement mock versions, and build an agent that uses them to handle common support scenarios.

3. **LangGraph workflow.** Build a LangGraph agent that handles the following workflow: (a) classify an incoming customer email, (b) route it to the appropriate specialist agent, (c) draft a response, (d) check the response for policy compliance, (e) send or revise. Include human-in-the-loop approval before sending.

4. **Multi-agent debate.** Implement a debate pattern where two agents argue for and against a given proposition. A third "judge" agent evaluates the arguments and determines the winner. Test on 5 controversial technical topics (e.g., "Microservices are always better than monoliths").

5. **Memory comparison.** Build an agent with and without episodic memory. Run it on 20 sequential tasks where later tasks are similar to earlier ones. Measure whether episodic memory improves performance on later tasks.

6. **MCP server.** Build an MCP server that exposes a SQLite database as a set of tools (list tables, describe table, execute query). Connect it to an MCP client and test natural language to SQL queries.

7. **Agent cost analysis.** Take any agent from this chapter and instrument it to track token usage and cost. Run it on 50 tasks and analyze the cost distribution. Identify which steps are most expensive and propose optimizations.

---

## References

Anthropic. (2024). Model Context Protocol specification. *Anthropic Documentation*. https://modelcontextprotocol.io

Chase, H. (2023). LangChain: Building applications with LLMs through composability. *GitHub Repository*. https://github.com/langchain-ai/langchain

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *Proceedings of the International Conference on Learning Representations*.

LangGraph. (2024). LangGraph: Build resilient language agents as graphs. *LangChain Documentation*. https://langchain-ai.github.io/langgraph/

Liu, X., Yu, H., Zhang, H., Xu, Y., Lei, X., Lai, H., ... & Dong, Y. (2023). AgentBench: Evaluating LLMs as agents. *Proceedings of the International Conference on Learning Representations*.

Mialon, G., Fourrier, C., Swift, C., Wolf, T., LeCun, Y., & Scialom, T. (2023). GAIA: A benchmark for general AI assistants. *arXiv preprint arXiv:2311.12983*.

OpenAI. (2024). Swarm: An experimental framework for lightweight multi-agent orchestration. *GitHub Repository*. https://github.com/openai/swarm

Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large language model connected with massive APIs. *arXiv preprint arXiv:2305.15334*.

Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., Van Den Driessche, G., ... & Hassabis, D. (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, 529(7587), 484–489.

Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., ... & Wang, C. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *Proceedings of the International Conference on Learning Representations*.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023b). Tree of thoughts: Deliberate problem solving with large language models. *Advances in Neural Information Processing Systems*, 36.

Zhou, S., Xu, F. F., Zhu, H., Zhou, X., Lo, R., Sridhar, A., ... & Neubig, G. (2024). WebArena: A realistic web environment for building autonomous agents. *Proceedings of the International Conference on Learning Representations*.
