from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()



def chain1():
    template = """Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    Question: {question}

    Helpful Answer:"""

    custom_prompt = PromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-4o-mini")

    simple_chain = (
        custom_prompt
        | model
        | StrOutputParser()
    )

    return simple_chain

# Example usage:
input_data = {
    "question": "What is quantum computing?"
}

chain1 = chain1()
result = chain1.invoke(input_data)
print("chain1",result)
# Quantum computing is a revolutionary technology that uses quantum bits (qubits) to perform calculations at speeds unattainable by classical computers. It leverages the principles of quantum mechanics, such as superposition and entanglement, to solve complex problems more efficiently. Thanks for asking!

def chain2():
    template = """Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    Question: {question}

    Helpful Answer:"""

    custom_prompt = PromptTemplate.from_template(template)

    model = ChatOpenAI(model="gpt-4o-mini")

    simple_chain = (
        custom_prompt
        | model
        | StrOutputParser()
    )
    result_chain = RunnablePassthrough.assign(
    answer=simple_chain
)

    return result_chain

# Example usage:
input_data = {
    "question": "What is quantum computing?"
}

chain2 = chain2()
result2 = chain2.invoke(input_data)
print(result2)

# {'question': 'What is quantum computing?', 'answer': 'Quantum computing is a type of computation that utilizes the principles of quantum mechanics to process information in fundamentally different ways than classical computers. It leverages quantum bits, or qubits, which can exist in multiple states simultaneously, enabling faster problem-solving for certain complex tasks. Thanks for asking!'}