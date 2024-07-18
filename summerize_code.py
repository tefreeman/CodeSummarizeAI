import os
from openai import OpenAI, Client
from embedding import EmbeddingModel


class ClientModel:
    def __init__(self, client: Client, model:str, embedding_model: EmbeddingModel):
        self.client = client
        self.model = model
        self.embedding_model = embedding_model

    def summarize_code(self, file_content, file_name, file_index, proj_files, proj_embeddings):
        relevant_contexts = self.embedding_model.get_relevant_context(file_content, file_name, file_index, proj_files, proj_embeddings)

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        summary = response.choices[0].message.content
        return summary

    def process_file(self, file_path, output_dir, file_index, proj_files, proj_embeddings, target):
    
        with open(file_path, 'r') as file:
            content = file.read()
        summary = self.summarize_code(content, os.path.basename(file_path), file_index, proj_files, proj_embeddings)
        
        rel_path = os.path.relpath(file_path, target)
        file_name, file_ext = os.path.splitext(rel_path)
        output_path = os.path.join(output_dir, f"{file_name}{file_ext}.summary.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as out_file:
            out_file.write(summary)
        
        print(f"Summary for {file_path} written to {output_path}")


    def process_folder(self, folder_path, output_dir):
        proj_files, proj_embeddings = self.embedding_model.embed_files(folder_path)
        
        for index, file_path in enumerate(proj_files):
            self.process_file(file_path, output_dir, index, proj_files, proj_embeddings, folder_path)

    def recursive_process_folder(self, folder_path, output_dir):
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
        proj_embeddings = [self.embedding_model.get_embedding(content) for content in proj_contents]
        
        # Now process files with embeddings
        for file_path in proj_files:
            self.process_file(file_path, output_dir, proj_embeddings, proj_files)

    def ask_questions(self, summary_directory):
        
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
            response = self.client.chat.completions.create(
                model=self.model,
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

    def summarize_project(self, summary_directory, question_answers):

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
            response = self.client.chat.completions.create(
                model=self.model,
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

