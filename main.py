import os
from pypdf import PdfReader
import openai
import chromadb
from chromadb.utils import embedding_functions
import random
from dotenv import load_dotenv

class PDFProcessor:
    def __init__(self, chromadb_path, openai_api_key, openai_model_name):
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name=openai_model_name)
        self.collection = self.client.get_or_create_collection("my_searchable_collection", embedding_function=self.openai_ef)
        self.openai_key = openai_api_key

    def save_data_to_chrom_db(self, text_chunks):
        try:
            for text in text_chunks:
                self.collection.add(documents=text.strip(), ids=[str(random.randint(1, 1000000))])
        except Exception as e:
            print('error:======', e)

    def search_data_from_chrom_db(self, text):
        try:
            search_result = self.collection.query(query_texts=[text], n_results=10)
            return search_result.get("documents")[0]
        except Exception as e:
            print('error:======', e)

    def pdf_handler(self, file_path):
        reader = PdfReader(file_path)
        page = reader.pages[0]
        text_line_by_line = []

        for i in range(len(reader.pages)):
            page = reader.pages[i]

            # extract the text line by line and save it in the array lines
            lines = page.extract_text().splitlines()
            filtered_lines = [item for item in lines if item.strip()]
            text_line_by_line.extend(filtered_lines)

        self.save_data_to_chrom_db(text_line_by_line)


    def handleSearch(self, search_text):
        search_result = self.search_data_from_chrom_db(search_text)
        
        prompt = f"consider these statements this are search results from the db {search_result}. \
                \n Please give me the related information that the search results have about {search_text} in a short and descriptive way as an answer to user and avoid additional sentences in the responce."

        openai.api_key = self.openai_key
        response = openai.ChatCompletion.create(
                                            model="gpt-3.5-turbo",
                                            messages=[{"role": "user", "content": prompt}]
                                            )
        return response.choices[0].message.content


# Usage example:
if __name__ == "__main__":
    load_dotenv()
    chromadb_path = os.getenv('CHORMADB_PATH')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_model_name = os.getenv('OPENAI_MODEL_NAME')

    pdf_processor = PDFProcessor(chromadb_path, openai_api_key, openai_model_name)
    # pdf in the current directory
    pdf_processor.pdf_handler("sample.pdf")
    search = pdf_processor.handleSearch("User search text")
    print("Search Result: ", search)

    


