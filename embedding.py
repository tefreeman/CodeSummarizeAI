import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from tree_sitter_language_pack import get_parser
import model_config
from openai import Client

class EmbeddingModel:
    def __init__(self, client: Client, model: str):
        self.client = client
        self.model = model

    @lru_cache(maxsize=None)
    def get_embedding_cached(self, text):
        return self.get_embedding(text)


    # TODO: Add support for more languages
    def extract_sections(self, content, file_extension):
        if file_extension in ['.py', '.pyw']:
            return self.extract_python_sections(content)
        elif file_extension in ['.ts', '.tsx']:
            return self.extract_typescript_sections(content)
        else:
            return [content]  # Return whole content if file type is unknown

    def extract_python_sections(self, content):
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


    def extract_typescript_sections(self, content):
        parser = get_parser('typescript')
        tree = parser.parse(bytes(content, "utf8"))
        def extract_nodes(node, acc):
            # ['class_declaration', 'function_declaration', 'method_definition']
            #if node.type in ['class_declaration', 'function_declaration', 'method_definition']:
            if True:
                acc.append(content[node.start_byte:node.end_byte])
            for child in node.children:
                extract_nodes(child, acc)
            return acc

        sections = extract_nodes(tree.root_node, [])
        return [section.strip() for section in sections if len(section.strip()) > 50]

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def get_relevant_context(self, file_content, file_name, file_index, proj_files, proj_embeddings, similarity_threshold=0.625, top_n=5):
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
            sections = self.extract_sections(similar_content, similar_file_extension)
            section_embeddings = [self.get_embedding_cached(section) for section in sections]

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


    def embed_files(self, folder_path:str):
        proj_files = []
        proj_embeddings = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.html', '.py')) or (file.endswith('.ts') and not file.endswith('.spec.ts')):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        content = f.read()
                    embedding = self.get_embedding(content)
                    proj_files.append(file_path)
                    proj_embeddings.append(embedding)
        
        return proj_files, proj_embeddings
    


