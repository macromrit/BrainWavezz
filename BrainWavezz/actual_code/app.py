# use -> pip install -r requirements.txt ## for downloading files 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI as OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle # for setting in api key
import openai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# faiss -> facebook ai 

class_11_maths = {
    "Sets and Functions": ("Sets", "Relations and Functions", "Trignometric Functions"),

    "Algebra": ("Principle of Mathematical Induction", "Complex Numbers anad Quadratic Equations", "Linear Inequalities", "Permutations and Combinations", "Binomial Theorem", "Sequence and Series"),

    "Coordinate Geometry": ("Straight Lines", "Conic Sections", 
    "Introduction to Three dimensional Geometry"),

    "Calculus": ("Limits and Derivatives",),

    "Mathematical Reasoning": ("Mathematical Reasoning",),

    "Statistics and Probablity": ("Statistics", "Probability")
}


# reading api key from api_key
with open("actual_code/api_key.txt", "r") as jammer:
    api_key = jammer.read().partition("=")[-1].strip()

# open ai key
openai.api_key = api_key

# process 1 ----------------------------------------------->

def pdfToTextConverter(file_path: str) -> tuple:
    """function to convert PDF file to text

    Args:
        file_path (str): path of target PDF

    Returns:
        tuple(bool, str):
                1) bool -> file is valid? True | False
                2) str -> if file valid -> text content, else error message
    """

    try: # try opening the file at given path
        pdf_reader = PdfReader(file_path)
    except: # if exceptions are raised in abv process
        return False, "Invalid PDF File / Readings weren't supported"
    else: # if file was opened and read successfully
        return True, ''.join(map(lambda page : page.extract_text(), pdf_reader.pages))


# process 2 ----------------------------------------------->

def compileMutiplePdfsGrade11():

    """
    stats:
        if grade 11 math pdfs are compiled and splitted token wise 
        provided token size per chunk is 1k
        698(~700) chunks are contrieved
    """


    # compiling all class 11 pdfs
    main_text = ""

    for i in range(1, 17):
        to_add = pdfToTextConverter("actual_code/cbse_11_chapters/chp{}.pdf".format(i))
        if to_add[0]: # if its a valid file
            main_text += to_add[1]
        else:
            print("invalid pdf found at -> cbse_11_chapters/chp{}.pdf".format(i))
    
    to_add = pdfToTextConverter("actual_code/cbse_11_chapters/solutions.pdf")
    
    if to_add[0]: # true -> valid file else invalid pdf
        main_text += to_add[1]
    else:
        print("invalid pdf found at -> cbse_11_chapters/solutions.pdf")

    return main_text


# process 3 ----------------------------------------------->

def langTextSplitter(pdf_content: str) -> list:
    """ function to split a given pdf file's text content
        into multiple chunks for easy matching and text processal
        !! this process saves time on searching customized data

    Args:
        pdf_content (str): _description_

    Returns:
        list: _description_
    """

    # splitter model being initialized
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1e3, # 1 * 10^3 = 1000
        chunk_overlap = 2e2, # 2 * 10^2 = 200 || chunk overlap's abt interconnecting splitted components
        length_function = len, # just passing Built-in Scope length function
    )

    # using abv model to split chunks
    chunks = text_splitter.split_text(text=pdf_content)

    return chunks


# process 4 ------------------------------------------------>

def embeddText(file_name: str, chunks: list = None):
    """
        chunks(list) -> text chunks which were splitted 

        if a pkl file already exists as a vector space for
        certain segment of data, its a crime to redo the 
        whole process again. as you can see for it
        tokens will be consumed in a rage pace

        if an exisiting file is called then we gotta read data in it
        else we have to write data to it
    """

    if os.path.exists(F"actual_code/embeddings/{file_name}.pkl") and not chunks:
        with open(F"actual_code/embeddings/{file_name}.pkl", "rb") as jammer:
            VectorStore = pickle.load(jammer)
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        VectorStore = FAISS.from_texts(chunks, embeddings) # lot of tokens are consumed in this process

        with open(F"actual_code/embeddings/{file_name}.pkl", "wb") as jammer:
            pickle.dump(VectorStore, jammer)
        
    return VectorStore


# process 5 ----------------------------------------------->

def poseQueries(query, VectorStore):
    assert query; # if query's True it passes else it won't an assertion error would be t hrown

    docs = VectorStore.similarity_search(query=query, k=4)

    llm = OpenAI(openai_api_key=api_key,temperature=.5, request_timeout=120)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    
    return response.strip().replace("\n", " ") # \n's are replace with spaces as questions generated would be stored in a text file, lines are source of queries to be fetched

# process 6 ----------------------------------------------->

def writeIntoText(line: str, filename: str):

    try:
        with open(filename, "a") as jammer:
            print(line, file=jammer)
    except:
        pass


# process 7 ----------------------------------------------->

def generateQuestion(chapter_name, question_list, vector_store):
    query = F"""
        generate a question from chapter {chapter_name}

        constraint 1: question generated must have an answer which is representable in a single line, don't give provement questions.

        constraint 2: make sure the question is unique and doesn't exist\
        before.

        constraint 3: ensure more than 3 different math concepts are to be\
        applied for solving the question.

        difficulty level of question: only expert mathematicians could solve. question shouldn't be too lengthy to solve.

        IMPORTANT !!! constraint 4: dont give simple questions (like 2+2 and more...)


        FORMAT OF OUTPUT - don't generate anything extra, just the question is fine:
            Question: <question generated>
    """

    return poseQueries(query, vector_store)



# process 9 -------------------------------------------------->
def genAns(question):

    # prompt base template
    template = """
    instructions:
        1. a math question will be given, analyze and be ready with resources for solving it.
        2. start solving the question, take your time and do it yourself from start, step-by-step. Don't predict values.
        3. after getting answer give output in format delimited by ```.
        TIP: atleast solve the question five times yourself and rectify and recheck before giving output.
        
    question:
        {question}

    output format:
        
        ```
question: 
<question must come here>

explanation: 
<step wise explanation>

final answer: 
<final answer>

        ```
    """

    # 
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    
    # step 5: assigning values to Prompt Template
    # print(prompt.format(question=question))
    
    # Step 6: Instantiate the LLMChain
    llm =  OpenAI(openai_api_key=api_key,model_name='gpt-3.5-turbo-0613',temperature=0, request_timeout=120)
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Step 7: Run the LLMChain
    output = chain.run(question)

    # processing output and extracting necessary data

    aoi = output.strip("```") 

    # splitting question
    holder1 = aoi.partition("explanation:")
    question_fed = holder1[0].partition("question:")[2]
    question_fed = question_fed.strip()
    
    # spltting explanation
    holder2 = holder1[2].partition("final answer:")
    explanation = holder2[0].strip()

    final_ans = holder2[2].strip().replace("```", "")

    return question_fed.replace("\n", "<br>"), explanation.replace("\n", "<br>"), final_ans.replace("\n", "<br>")


# process 8 ------------------------------------------------>

def main_grade_11_math(questions_per_chapter: int):

    # reading content in vector store for training data
    VectorStore = embeddText("grade_11_math")

    # read the list of questions premade
    with open("actual_code/question_list.txt", "r") as jammer:
        pre_made_questions = jammer.readlines()

    chapter_set = """
Sets
Relations and Functions
Trignometric Functions
Principle of Mathematical Induction
Complex Numbers anad Quadratic Equations
Linear Inequalities
Permutations and Combinations
Binomial Theorem
Sequence and Series
Straight Lines
Conic Sections
Introduction to Three dimensional Geometry
Limits and Derivatives
Mathematical Reasoning
Statistics
Probability
    """.strip().split("\n")
    
    temp_query_list = list()

    main_return_list = list()

    for i in chapter_set:

        for _ in range(questions_per_chapter):

            question_generated = generateQuestion(i, 
                                                  [], 
                                                  VectorStore)
            # print(question_generated)
            temp_query_list.append(question_generated)

            question = question_generated.partition(":")[2].strip()

            # answer is generated and returned as question , steps to solve and answer in a list
            q_a_pack = genAns(question)

            main_return_list.append(q_a_pack)


    # write down newly generated questions to question_list.txt
    for question in temp_query_list:
        writeIntoText(question, "actual_code/question_list.txt")

    return main_return_list















# code to tokenize grade 11 cbse math textbook pdfs -------->

# total_text = compileMutiplePdfsGrade11()
    # text_splitter = langTextSplitter(total_text)
    # embedding the abv text made for grade 11 math and uploading 
    # them to a pkl file names grade_11_math in embeddings directory
    # VectorStore = embeddText("grade_11_math")

# ---------------------------------------------------------->

if __name__== "__main__":
    

    pass    


    x = main_grade_11_math(1)

    cnt = 1

    for question, steps, final_ans in x:

        print("="*40) 
        print("\n\n")

        print("QUESTION:")
        print(question, end="\n\n\n")

        print("EXPLANATION:")
        print(steps, end="\n\n\n")

        print("FINAL ANSWER:")
        print(final_ans, end="\n\n\n")

        print("\n\n")
        print("="*40)
        











#     user_input = input("Enter a concept: ")    

#     template = """

# {question}
# solve the above question from start and find the solution yourself, take your time
# dont give wrong answers
# """

#     template = """
        
# respond back to the question : {question}

#         """

#     # Step 4: Define the Prompt Template
#     prompt = PromptTemplate(
#         input_variables=["question"],
#         template=template,
#     )
    
#     # step 5: assigning values to Prompt Template
#     print(prompt.format(question=user_input))
    
#     # Step 6: Instantiate the LLMChain
#     llm =  OpenAI(openai_api_key=api_key,model_name='gpt-3.5-turbo-0613',temperature=0)
#     chain = LLMChain(llm=llm, prompt=prompt)
    
#     # Step 7: Run the LLMChain
#     output = chain.run(user_input)
#     print(output)


    # 
    # # print(main_grade_11_JEE_math(1))

    # llm = OpenAI(openai_api_key=api_key,model_name='text-davinci-003',temperature=0)

    # prompt_template = PromptTemplate(input_variables=["SEE"], template="Say {SEE} to me")
    
    # # Step 5: Print the Prompt Template
    # print(prompt_template.format(concept="HELLO"))
    
    # Step 6: Instantiate the LLMChain
    # chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # # Step 7: Run the LLMChain
    # output = chain.run("hello")
    # print(output)

#     llm_chain = LLMChain(
# prompt="Hey what day is it today?",
# llm=llm)
#     print(llm_chain.run())

    # print(generateQuestion("Relations and Functions", ["find probability of finding a yellow out of a sack containing 3 red balls and 9 yellow balls"], VectorStore))

    # print(poseQueries(query, VectorStore))

    # items = list(class_11_maths.values())
    # print('\n'.join(map(lambda x : '\n'.join(x),items)))