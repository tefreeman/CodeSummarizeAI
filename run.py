from openai import OpenAI
from summerize_code import ClientModel
from embedding import EmbeddingModel

if __name__ == "__main__":
    target = "D:\\repos\\ycrawl\\src"
    output_dir = "code_summaries_2"
    

    embedding_model = EmbeddingModel(OpenAI(base_url="", api_key="lm-studio"), "CompendiumLabs/bge-large-en-v1.5-gguf")
    
    
    f = open("test.ts", "r")

    text = f.read()
    
    
    res = embedding_model.extract_sections(text, '.ts')
    print(res)

    # local_client = ClientModel(client=OpenAI(base_url="", api_key="lm-studio"), model="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",  embedding_model=embedding_model)

    # local_client.process_folder(target, output_dir)
    
    # input("Press Enter to start summarizing code...")

    # question_answers = local_client.ask_questions(output_dir)
    # if question_answers:
    #     proj_summary = local_client.summarize_project(output_dir, question_answers)
    #     print("Project Summary:")
    #     print(proj_summary)
        
    #     with open('proj_summary.txt', 'w') as f:
    #         f.write(proj_summary)
