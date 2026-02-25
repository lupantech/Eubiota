import os
import numpy as np
import openai

from dotenv import load_dotenv
load_dotenv()


SUMMARIZE_PROMPT_TEMPLATE = """
You are an expert AI assistant. Your task is to provide a clear, concise, and accurate answer to the user's query based **exclusively** on the provided reference information.

## Step-by-Step Instructions
1.  **Analyze the Query:** First, fully understand the user's query and identify the specific information being asked for.
2.  **Scan for Relevance:** Read through each numbered chunk in the reference information. Identify all chunks that contain information directly relevant to answering the query. A simple keyword match is not sufficient; the chunk must contain a substantive fact that helps answer the question.
3.  **Extract Key Facts & Synthesize:** From the relevant chunks, extract only the key facts and figures needed. Synthesize these extracted facts into a comprehensive, single-paragraph answer. Write the answer in your own words. **Do not** copy entire chunks.

## Output Format and Example

**IMPORTANT:** You must follow this format exactly.

### Example Input
- **User Query:** What were the key financial results for Q4 2023?
- **Reference Information:**
[1] The company's new "Project Starlight" initiative launched in January 2024.
[2] In Q4 2023, the company reported a total revenue of $5.2 million and a net profit of $800,000. This was a 15% increase in revenue compared to Q3 2023.
[3] Marketing spend in Q4 2023 was focused on digital channels, totaling $450,000.
[4] The CEO stated that the strong Q4 performance was driven by robust sales in the North American market.

### Example Output
Answer:
In the fourth quarter of 2023, the company achieved a total revenue of $5.2 million, which represented a 15% increase from the previous quarter, and a net profit of $800,000. The strong performance was attributed to robust sales in the North American market. The marketing expenditure for this period was $450,000.

---
## Your Turn

### User Query
{query}

### Reference Information
{reference_information}

### Output
"""

class RAG_Model():
    def __init__(self, llm, chunk_size=200, chunk_overlap=20, top_k=10, embeddings_model="text-embedding-3-large"):
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embeddings_model = embeddings_model

    def _chunk_content(self, content):
        """
        Chunks the website content into smaller chunks based on the chunk size and overlap.
        Parameters:
            content (str): The website content to chunk.
        Returns:
            list: A list of chunks.
        """
        # Split the content string by whitespace characters
        words = content.split()
        ptr = 0
        chunks = []
        while True:
            start, end = ptr, min(ptr + self.chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            ptr = end - self.chunk_overlap
        return chunks

    def _embed_strings(self, strings):
        """
        Embed the strings using OpenAI's embedding model.
        Parameters:
            strings (list): A list of strings to embed.
        Returns:
            list: A list of embeddings.
        """
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            embeddings = client.embeddings.create(
                input=strings,
                model=self.embeddings_model
            )
            res = [embedding.embedding for embedding in embeddings.data]
            return res
        except Exception as e:
            raise Exception(f"Error embedding strings: {str(e)}")

    def _cosine_similarity(self, a, b):
        """
        Calculate the cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _rank_chunks(self, query_embedding, chunk_embeddings):
        """
        Rank the chunks based on the query embedding.
        Parameters:
            query_embedding (list): The embedding of the query.
            chunk_embeddings (list): The embeddings of the chunks.
        Returns:
            list: The indices of the ranked chunks in descending order of similarity.
        """
        similarities = [self._cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
        return list(np.argsort(similarities)[::-1])

    def _concatenate_chunks(self, chunks):
        """
        Concatenate the chunks into a single string.
        """
        for i, chunk in enumerate(chunks):
            chunks[i] = f"Chunk [{i+1}]\n{chunk}"
        return "\n".join(chunks)

    def _construct_final_output(self, query, reference_information):
        """
        Construct the final output from the top chunks.
        """
        summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            query=query,
            reference_information=reference_information
        )
        summary = self.llm.generate(summary_prompt)
        return summary

    def generate_summary_from_content(self, query, content):
        # step 1: chunk the content
        chunks = self._chunk_content(content)

        # step 2: embed the chunks
        embeddings = self._embed_strings([query] + chunks)
        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]

        # step 3: rank the chunks
        ranked_chunks = self._rank_chunks(query_embedding, chunk_embeddings)
        top_chunks = [chunks[i] for i in ranked_chunks[:self.top_k]]

        # step 4: summarize the top chunks
        reference_string = self._concatenate_chunks(top_chunks)
        summary = self._construct_final_output(query, reference_string)

        return summary
