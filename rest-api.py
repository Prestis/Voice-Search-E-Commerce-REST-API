#from openai import OpenAI
from flask import Flask, request, jsonify
import whisper
#from faster_whisper import WhisperModel
from dotenv import load_dotenv
#import spacy
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import tempfile
 
import os 


load_dotenv()

token_count = 0

app = Flask(__name__)


#Langchain prompt initiation
template = """
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Remember to use left joins to get the info only from the 'products' table.
Question: {question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)

##Initializing database connection for langchain
db_uri = os.getenv('DB_URI')
db = SQLDatabase.from_uri(db_uri)
def get_schema(_):
    return db.get_table_info()

##Initializing the LLM for langchain
llm = ChatOpenAI(
    model = "gpt-3.5-turbo"
)

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    |prompt
    |llm.bind(stop="\nSQL Result:")
    |StrOutputParser()
)

@app.after_request
def add_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = "Content-Type, Access-Control-Allow-Headers, Authorization"
    response.headers['Access-Control-Allow-Methods']= "POST, GET, PUT, DELETE, OPTIONS" 
    return response



@app.route(rule="/transcriber",methods = ['POST'])
def get_transcribed_audio():
    #print(request.files['audio'])


    if 'audio' not in request.files:
        return 'No file part', 400

    file = request.files['audio']
    if file.filename == '':
        return 'No selected file', 400

    # Ensure the file has the correct extension, e.g., .mp3
    if not file.filename.endswith('.mp3'):
        return jsonify({'error': 'Invalid file type. Only MP3 files are supported.'}), 400
    
    # Get the temporary directory path
    temp_dir = tempfile.gettempdir()

    # Construct the file path using the correct temp directory
    file_path = os.path.join(temp_dir, file.filename)

    try:
        # Save the file temporarily
        file.save(file_path)
        print('File Saved')
        # Process the file as needed
        # For example, you might analyze the audio file here
        model = whisper.load_model("base")
        result = model.transcribe(file_path,fp16=False)
        # result = model.transcribe("Corsair-query.mp3",fp16=False)
        response = result["text"]
        print(response)
        return sql_chain.invoke({"question": response })

        #return jsonify({'message': 'File processed successfully'})
        #return 'File processed successfully', 200

    finally:
        # Clean up: remove the file after processing
        print('Cleaning up the path ' + file_path)
        if os.path.exists(file_path):
            os.remove(file_path)

    # # Process the file as needed
    # # ...

    # # Clean up the file
    # os.remove(file_path)

    

    

    # return f'File received: {blob_url}', 200
    # if 'audio_data' not in request.files:
    #     return 'No file part ', 400
    
    # file = request.files['audio_data']
    # if file.filename == '':
    #     return 'No selected file', 400
    
    # return f'File received: {file.filename}', 200

    # model = whisper.load_model("base")
    # result = model.transcribe(url,fp16=False)
    # # result = model.transcribe("Corsair-query.mp3",fp16=False)
    # response = result["text"]
    # print(response)
    # return sql_chain.invoke({"question": response })

    #faster_model = WhisperModel(model_size,device="cpu",compute_type="int8")
    #faster_result, _ = faster_model.transcribe("Corsair-query.mp3",beam_size='5')
    #faster_result = list(faster_result)

    # doc = nlp(result["text"])
    # for ent in doc.ents:
    #     print(ent.text, "|",ent.label_)

    #response = personal_assistant(response)
    #return faster_result
    #return response

if __name__ == "__main__":
    #Setting host as localhost
    app.run(host= "0.0.0.0", debug=True)


