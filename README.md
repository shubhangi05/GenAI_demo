---

Code demonstrates a multi-part **LLM-powered system** combining chat, agents, and retrieval-based reasoning.

1. It implements a **simple chat session with an LLM** using **Gemini** (free tier). The model is intentionally configured to respond rudely, showcasing how system instructions can control tone and behavior. This setup requires an API key.

2. It includes a **basic agent with multiple tools**, such as:

* summing two numbers
* checking if a number is prime
* fetching the value of Bitcoin

This demonstrates tool usage within an LLM-driven workflow, again relying on an API key.

3. Finally, it builds a **RAG (Retrieval-Augmented Generation) system** over *The Housemaid* novel series, allowing users to ask questions about the story (e.g., about Millie). The system uses:

* **Hugging Face embeddings** (local, no external API calls) for vector representation -- Requires no API key
* **Pinecone** for storing and retrieving document chunks -- Require API key
* **Ollama (LLaMA)** for the `transform_query` step, which rewrites user queries into standalone questions before embedding -- Requires no API key

Overall, the project showcases how to combine **LLMs, agents, and RAG pipelines** using both cloud-based and fully local components.
