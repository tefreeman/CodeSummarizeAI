import os
import openai
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from tree_sitter_language_pack import get_parser

# TODO: refactor out of global scope
local_client = None


# Huge speed up by caching embeddings
@lru_cache(maxsize=None)
def get_embedding_cached(text, model="CompendiumLabs/bge-large-en-v1.5-gguf"):
    return get_embedding(text, model)


# TODO: Add support for more languages
def extract_sections(content, file_extension):
    if file_extension in ['.py', '.pyw']:
        return extract_python_sections(content)
    elif file_extension in ['.ts', '.tsx']:
        return extract_typescript_sections(content)
    else:
        return [content]  # Return whole content if file type is unknown

def extract_python_sections(content):
    parser = get_parser('python')
    tree = parser.parse(bytes(content, "utf8"))
    
    def extract_nodes(node, acc):
        if node.type in ['class_definition', 'function_definition', 'decorator', 'import_statement', 'import_from_statement']:
            acc.append(content[node.start_byte:node.end_byte])
        for child in node.children:
            extract_nodes(child, acc)
        return acc

    sections = extract_nodes(tree.root_node, [])
    return [section.strip() for section in sections if len(section.strip()) > 50]

def extract_typescript_sections(content):
    parser = get_parser('typescript')
    tree = parser.parse(bytes(content, "utf8"))
    def extract_nodes(node, acc):
        if node.type in ['class_declaration', 'function_declaration', 'method_definition']:
            acc.append(content[node.start_byte:node.end_byte])
        for child in node.children:
            extract_nodes(child, acc)
        return acc

    sections = extract_nodes(tree.root_node, [])
    return [section.strip() for section in sections if len(section.strip()) > 50]

def get_embedding(text, model="CompendiumLabs/bge-large-en-v1.5-gguf"):
    text = text.replace("\n", " ")
    return local_client.embeddings.create(input=[text], model=model).data[0].embedding

def get_relevant_context(file_content, file_name, file_index, proj_files, proj_embeddings, similarity_threshold=0.625, top_n=5):
    try:
        file_embedding = proj_embeddings[file_index]
    except IndexError:
        print(f"Error: file_index {file_index} is out of range for proj_embeddings")
        return []

    if isinstance(file_embedding, list):
        file_embedding = np.array(file_embedding).reshape(1, -1)

    file_extension = os.path.splitext(file_name)[1]
    
    relevant_contexts = []
    for i, similar_file in enumerate(proj_files):
        if i == file_index:
            continue  # Skip the current file

        try:
            with open(similar_file, 'r') as f:
                similar_content = f.read()
        except Exception as e:
            print(f"Error reading file {similar_file}: {str(e)}")
            continue

        similar_file_extension = os.path.splitext(similar_file)[1]
        sections = extract_sections(similar_content, similar_file_extension)
        section_embeddings = [get_embedding_cached(section) for section in sections]

        section_embeddings = [np.array(se) if isinstance(se, list) else se for se in section_embeddings]
        if len(section_embeddings) == 0:
            continue
        section_embeddings = np.array(section_embeddings)

        try:
            similarities = cosine_similarity(file_embedding, section_embeddings)[0]
        except ValueError as ve:
            print(f"Error in processing similarities: {ve}")
            continue
        
        top_indices = similarities.argsort()[-(top_n * 2):][::-1]  # Initial broader selection
        for idx in top_indices:
            if similarities[idx] > similarity_threshold:
                relevant_contexts.append({
                    'file': os.path.basename(similar_file),
                    'section': sections[idx],
                    'similarity': similarities[idx]
                })

    relevant_contexts.sort(key=lambda x: x['similarity'], reverse=True)
    return relevant_contexts[:top_n]

def embed_files(folder_path):
    proj_files = []
    proj_embeddings = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.html', '.py')) or (file.endswith('.ts') and not file.endswith('.spec.ts')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                embedding = get_embedding(content)
                proj_files.append(file_path)
                proj_embeddings.append(embedding)
    
    return proj_files, proj_embeddings

def summarize_code(file_content, file_name, file_index, proj_files, proj_embeddings):
    relevant_contexts = get_relevant_context(file_content, file_name, file_index, proj_files, proj_embeddings)

    context_str = "\n\n".join([f"From {ctx['file']} (similarity: {ctx['similarity']:.2f}):\n{ctx['section']}" 
                                for ctx in relevant_contexts])
    
    prompt = f"""Summarize the file: {file_name}
    Do not add any commentary or opinion, just summarize the code content concisely.
    Your're goal is to provide a clear and informative summary of the code content while excluding any irrelevant information.
    
    Code content:
    {file_content}

    This file is part of a larger project. Here is some extra context, do not include this in the summary:
    {context_str}
    """

    response = local_client.chat.completions.create(
        model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )
    summary = response.choices[0].message.content
    return summary

def process_file(file_path, output_dir, file_index, proj_files, proj_embeddings):
 
    with open(file_path, 'r') as file:
        content = file.read()
    summary = summarize_code(content, os.path.basename(file_path), file_index, proj_files, proj_embeddings)
    
    rel_path = os.path.relpath(file_path, target)
    file_name, file_ext = os.path.splitext(rel_path)
    output_path = os.path.join(output_dir, f"{file_name}{file_ext}.summary.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as out_file:
        out_file.write(summary)
    
    print(f"Summary for {file_path} written to {output_path}")


def process_folder(folder_path, output_dir):
    proj_files, proj_embeddings = embed_files(folder_path)
    
    for index, file_path in enumerate(proj_files):
        process_file(file_path, output_dir, index, proj_files, proj_embeddings)

def recursive_process_folder(folder_path, output_dir):
    proj_files = []
    proj_contents = []
    # First pass: collect all file contents and paths
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.html', '.py')) or (file.endswith('.ts') and not file.endswith('.spec.ts')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                proj_files.append(file_path)
                proj_contents.append(content)
    
    # Generate embeddings for all files
    proj_embeddings = [get_embedding(content) for content in proj_contents]
    
    # Now process files with embeddings
    for file_path in proj_files:
        process_file(file_path, output_dir, proj_embeddings, proj_files)

def ask_questions(summary_directory, client):
    
    all_summaries = ""
    for root, dirs, files in os.walk(summary_directory):
        for file in files:
            if file.endswith('.summary.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    all_summaries += f"File: {file}\n{f.read()}\n\n"

    questions_prompt = f"""Based on the following summaries of individual files in a project, ask questions to help you better understand the project:
    {all_summaries}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert software architect, skilled at understanding and summarizing complex projects."},
                {"role": "user", "content": questions_prompt}
            ],
            temperature=0.65,
        )
        questions = response.choices[0].message.content
        
        with open('proj_questions.txt', 'w') as f:
            f.write(questions)

        print("Project Questions:")
        print(questions)
        print("Please answer the questions in proj_questions.txt and press Enter to continue...")
        input()

        with open('proj_questions.txt', 'r') as f:
            question_answers = f.read()
        
        return question_answers
    
    except Exception as e:
        print(f"Error in generating questions: {str(e)}")
        return None

def summarize_project(summary_directory, question_answers, client):

    all_summaries = ""
    for root, dirs, files in os.walk(summary_directory):
        for file in files:
            if file.endswith('.summary.txt'):
                with open(os.path.join(root, file), 'r') as f:
                    all_summaries += f"File: {file}\n{f.read()}\n\n"

    proj_summary_prompt = f"""Based on the following summaries of individual files in a project, provide a comprehensive github readme overview of the entire project.

    Here are some questions and answers to help you better understand the project:
    {question_answers}

    Summaries of individual files in the project:
    {all_summaries}

    Please include in your project overview:
    1. The main purpose and functionality of the project
    2. Key components or modules and their interactions
    3. Overall architecture and design patterns used
    4. Important technologies or frameworks employed
    5. Any notable features or algorithms implemented

    Provide a informative summary that gives a clear understanding of the project's structure and functionality."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert software architect, skilled at understanding and summarizing complex projects."},
                {"role": "user", "content": proj_summary_prompt}
            ],
            temperature=0.65,
        )
        proj_summary = response.choices[0].message.content
        return proj_summary
    except Exception as e:
        return f"Error summarizing project: {str(e)}"

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
