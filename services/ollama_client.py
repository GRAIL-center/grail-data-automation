from ollama import chat


class OllamaClient:
    """
    Ollama client for embedding and generation.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        embed_model: str = "mxbai-embed-large",
        temperature: float = 0.2,
    ):
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature

    def summarize(self, text: str, context: str = "") -> str:
        """
        Summarizes text using the Ollama client.

        Parameters
        ----------
        text : str
            The text to summarize.
        context : str
            The context of the document.

        Returns
        -------
        str
            The summarized text.
        """
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the following document. Only write the summary. Do not include any additional information. Context: {context} Text to summarize: {text}",
                }
            ],
            options={"temperature": self.temperature},
        )
        return response.message.content
