"""
Test script to verify LangChain installation and functionality
"""
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def test_langchain_functionality():
    """
    Simple test to verify LangChain is working properly
    Note: This is just a test and requires an OpenAI API key to run
    """
    print("Testing LangChain functionality...")
    
    # This is just a test template - no actual API calls will be made without a key
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    
    # This would require an API key to actually run
    # model = ChatOpenAI(temperature=0.7)
    # chain = prompt | model | StrOutputParser()
    # result = chain.invoke({"topic": "programming"})
    # print(result)
    
    print("LangChain imports working correctly!")
    print("Note: Actual API calls require an OpenAI API key")

if __name__ == "__main__":
    print("LangChain and related packages are installed!")
    test_langchain_functionality()
