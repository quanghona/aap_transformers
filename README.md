# aap_transformers

Hugging Face Transformers integration for the Agent Design Pattern framework.

![PyPI - License](https://img.shields.io/pypi/l/aap_transformers)
![PyPI - Downloads](https://img.shields.io/pypi/dm/aap_transformers)
![PyPI - Version](https://img.shields.io/pypi/v/aap_transformers?label=%20)
![Python Version](https://img.shields.io/pypi/pyversions/aap_transformers)

## What is this?

`aap_transformers` provides Hugging Face Transformers-specific implementations of the core agent abstractions defined in [`aap_core`](../core/). It enables you to run open-weight LLMs locally (or on cloud GPUs) within the agent orchestration framework, supporting causal language models with system prompts, user templates, and multi-turn conversations — all without relying on external API providers.

## Quick Start

```bash
pip install aap_transformers
```

```python
from aap_transformers.chain import ChatCausalMultiTurnsChain

chain = ChatCausalMultiTurnsChain(
    model="HuggingFaceTB/SmolLM3-3B",
    system_prompt="You are a helpful assistant.",
    user_prompt_template="{query}",
    device="cuda",
    max_new_tokens=2048,
)
```

## Features

- **`ChatCausalMultiTurnsChain`**: A Transformers-backed multi-turn chain that extends `BaseCausalMultiTurnsChain` from `aap_core`. Supports system prompts, user templates, device placement (CPU/CUDA), and configurable generation parameters.
- **Local model inference**: Run open-weight models directly via the `transformers` library — no external API keys or cloud services required.
- **GPU acceleration**: Native support for CUDA devices and `accelerate`-based model loading for efficient inference.
- **Token usage tracking**: Integrates with `aap_core`'s `TokenUsage` type for consistent tracking across agents.

## Installation

### Core dependency

`aap_transformers` depends on `aap_core`. Install both together:

```bash
pip install aap_core aap_transformers
```

### Optional dependencies

```bash
# Weaviate integration
pip install "aap_transformers[weaviate]"
```

### Development setup

```bash
uv sync --group dev
```

## Usage

### Building an agent with a Transformers model

```python
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from aap_core.agent import BaseAgent
from aap_core.types import AgentMessage
from aap_transformers.chain import ChatCausalMultiTurnsChain

class MyAgent(BaseAgent):
    chain: BaseLLMChain

    def execute(self, message: AgentMessage, **kwargs) -> AgentMessage:
        self.state = "running"
        message = self.chain.invoke(message, **kwargs)
        message.execution_result = "success"
        message.origin = self.card.name
        self.state = "idle"
        return message

chain = ChatCausalMultiTurnsChain(
    model="ibm-granite/granite-4.0-h-1b",
    system_prompt="You are a helpful coding assistant.",
    user_prompt_template="{query}",
    device="cuda",
    max_new_tokens=4096,
)

agent = MyAgent(
    chain=chain,
    card=AgentCard(
        name="coding-agent",
        description="A coding assistant agent",
        skills=[AgentSkill(id="coding", name="Coding", description="Writes code")],
        capabilities=AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        url="localhost",
        version="0.1.0",
    ),
)

result = agent.execute(AgentMessage(query="Write a function to compute Fibonacci numbers."))
```

### Multi-agent sequential pipeline

```python
from aap_core.orchestration import SequentialAgent
from aap_transformers.chain import ChatCausalMultiTurnsChain

code_chain = ChatCausalMultiTurnsChain(
    model="swiss-ai/Apertus-8B-Instruct-2509",
    system_prompt="You are a helpful coding assistant. Write only the function implementation.",
    user_prompt_template="{query}",
    device="cuda",
    max_new_tokens=8192,
)
code_chain.final_response_as_context("code")

doc_chain = ChatCausalMultiTurnsChain(
    model="HuggingFaceTB/SmolLM3-3B",
    system_prompt="You are a developer advocate. Write docstrings for the provided function.",
    user_prompt_template="Input query: {query}\n\nFunction: {context_code}",
    device="cuda",
    max_new_tokens=8192,
)

sequential_agent = SequentialAgent(
    agents=[code_agent, doc_agent],
    card=AgentCard(name="pipeline", description="Sequential code-doc pipeline"),
)
```

## Architecture

`aap_transformers` is a subpackage of the [Agent Design Pattern](https://github.com/quanghona/agent_design_pattern) monorepo. It sits on top of `aap_core` and provides Hugging Face Transformers-specific implementations:

```
aap_core          ← Core agent abstractions (BaseAgent, BaseLLMChain, BaseRetriever, AgentMessage)
    └── aap_transformers  ← Transformers integrations
            ├── chain.py      ChatCausalMultiTurnsChain
            └── retriever.py  RetrieverAdapter (planned)
```

| Package | Description |
|---------|-------------|
| `aap_core` | Core orchestration framework with agent, chain, and retriever abstractions |
| `aap_langchain` | LangChain integration (cloud/local chat models) |
| `aap_llamaindex` | LlamaIndex integration |
| `aap_dspy` | DSPy integration |
| `aap_transformers` | Hugging Face Transformers integration (this package) |

## Documentation

- Example notebooks: [`example/transformers/`](../../example/transformers/)
  - [Simple Loop](../../example/transformers/simple_loop.ipynb)
  - [Sequential](../../example/transformers/sequential.ipynb)
  - [Coordinator](../../example/transformers/coordinator.ipynb)
  - [Self-Reflection](../../example/transformers/self_reflection.ipynb)
  - [Debate](../../example/transformers/debate.ipynb)
  - [Voting](../../example/transformers/voting.ipynb)
  - [Parallel](../../example/transformers/parallel.ipynb)
  - [Cross-Reflection](../../example/transformers/cross_reflection.ipynb)

## Contributing

See the [project contributing guide](../../.github/instructions/project-common-core.instructions.md) for details on development setup, code style, and testing.

## License

This project is licensed under the MIT License — see the [LICENSE](../../LICENSE) file for details.
