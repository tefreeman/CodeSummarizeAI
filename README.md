# Code Summarizer Project

This project is designed to automatically analyze, embed, and summarize code files within a specified project directory. It leverages NLP models to generate meaningful summaries for individual files and subsequently constructs an overall summary of the project.

## Features

- **Section Extraction:** Extracts relevant sections from Python and TypeScript files (e.g., class definitions, function definitions).
- **Embedding Computation:** Creates embeddings for code sections and files using specified AI models.
- **Contextual Similarity:** Determines the most relevant contexts within the project using cosine similarity of embeddings.
- **Summarization:** Generates comprehensive summaries for individual code files.
- **Project Overview:** Provides a high-level overview and summary of the entire project, mimicking a comprehensive GitHub README.

## Project Workflow

1. **File and Section Extraction:**

   - Files with extensions `.html`, `.py`, and `.ts` (excluding `.spec.ts`) are scanned within the specified directory.
   - Python and TypeScript files have their relevant sections extracted using Tree-sitter-based parsers.

2. **Embedding and Similarity Calculation:**

   - Text embeddings are calculated for each file and its sections using cached embeddings for efficiency.
   - Cosine similarity is computed to identify and prioritize the most relevant sections across files.

3. **Summarization of Individual Files:**

   - Each file is summarized using GPT-based local models, incorporating additional relevant contexts from the project.

4. **Question Generation and Answering:**

   - Summaries of individual files are used to generate questions to deepen the understanding of the project.
   - Users can provide answers to these generated questions to enrich the final project summary.

5. **Project Summarization:**
   - Based on the individual file summaries and user-provided answers, a comprehensive project summary is generated to serve as a detailed README.

## Dependencies

- `os`: for directory and file handling
- `openai`: for embedding and chat completion with AI models
- `numpy`: for handling numerical operations
- `sklearn`: for computing cosine similarity
- `functools`: for caching function results
- `tree_sitter_language_pack`: a custom library for Tree-sitter-based language parsing

## Setup

1. **Install Dependencies:**
   Make sure you install the necessary libraries via pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download and Configure AI Models:**
   Ensure that the required models for embedding and summarization are available and configured appropriately. This may involve setting up API keys and endpoints for OpenAI and local models.

3. **Configure Paths:**
   Edit the `target` variable to point to the project directory you want to analyze and `output_dir` for storing summarized outputs.

4. **Setup API Keys:**
   Replace the placeholders in the script with your actual OpenAI API keys and endpoints.

## Running the Project

1. **Process Folder:**
   Run the script to process the target folder and generate summaries.

   ```bash
   python summerize_code.py
   ```

2. **Generate and Answer Questions:**
   The script will prompt you to answer project-related questions. Provide your answers as instructed.

3. **Generate Project Summary:**
   After answering the questions, the script will generate a high-level project summary and save it to `proj_summary.txt`.

## Usage

To use the summarization functionality within another script or for different projects:

1. Import the necessary functions from `summerize_code.py`.
2. Call `process_folder` to generate summaries for a new project directory.
3. Follow the prompts to generate a comprehensive project overview.

## Example

In the script's main section, you can see an example usage:

```python
if __name__ == "__main__":
    target = "C:\\Users\\Trevor\\Documents\\earlyCodingAdventures\\foodApp\\ionic_app\\src\\app"
    output_dir = "code_summaries_2"

    # OpenAI ChatGPT better at summarizing code
    client = openai.OpenAI(
        organization='',
        project='',
        api_key=''
    )

    # Local model for generating summaries of text
    local_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    process_folder(target, output_dir)

    input("Press Enter to start summarizing code...")

    question_answers = ask_questions(output_dir, client)
    if question_answers:
        proj_summary = summarize_project(output_dir, question_answers, client)
        print("Project Summary:")
        print(proj_summary)

        with open('proj_summary.txt', 'w') as f:
            f.write(proj_summary)
```

## Contributing

Feel free to open issues or submit pull requests to enhance the functionality or fix any bugs. Contributions are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
