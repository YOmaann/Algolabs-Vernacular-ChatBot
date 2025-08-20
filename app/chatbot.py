# from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import InMemorySaver # recommended to use (Async)PostgresSaver for production capabilities
from langchain_core.tools import BaseTool
from langchain_core.tools import tool
from typing import Annotated, List, TypedDict
from langgraph.prebuilt import create_react_agent as create_langgraph_react_agent
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, RemoveMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import ChatPromptTemplate
import time
import os
from dotenv import load_dotenv
load_dotenv()
import re

# Syntactic but not semantic
# from flashtext import KeywordProcessor


def extract_raw_xml(response: str) -> str:
    return re.sub(r"^```xml\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)


# For google models
from langchain_google_genai import ChatGoogleGenerativeAI

# For xml parsing
import xml.etree.ElementTree as ET

# Geocoding Tool
from tools.geocoding import get_geo_tool


class ChatBotState(TypedDict):
    """State schema for the chatbot"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    keyword: str
    is_valid: bool  
    questions: List[str]
    is_waiting: bool
    reference_q: str
    reference_sum: str
    loop_thread: List[str]

class MyHelpfulBot():
    def __init__(self, model="gemini-2.5-flash", persist_directory="podak"):
        # self.llm = ChatGroq(
        #     model=model,
        #     temperature=0.4,
        #     max_tokens=None,
        #     reasoning_format="parsed",
        #     timeout=None,
        #     max_retries=2,
        # )

        
        self.embedfn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        keyword = ['Shared Visions & Goals', 'Vision & Mission', 'Culture and Employee Attraction in Financial Services', 'Compliance and Trust', "Citizen's Charter for Banks", 'Confidentiality & Usage Restrictions', 'KYC & AML Procedures', 'Identity Verification & Compliance', 'Financial Products & Services', 'Banking Services', 'Banking Service Operations', 'Customer Remedies', 'Customer Service Policies & Ombudsman Scheme Publicization', 'Customer Service Initiatives', 'Bank Website Access Policy', 'Bank Website Access', 'Bank Compensation Policy Access', 'Kotak Bank Security Repossession Policy Access', 'Bank Privacy Policy', 'Privacy Policy', 'Privacy Commitments', 'Consent for Information Disclosure', 'Legal Protections & Defenses', 'Consent and Disclosure', 'Third Party Support Services', 'Privacy Policy & Charter Access', 'Bank Service Standards', 'Services for Vulnerable Groups', 'Banking Facilities for Disabled', 'Bank Death Claim Process & Info Link', 'Bank Communication & Regulations', 'Bank Services', 'Locker Facilities Guide', 'SA/CA Account Safe Deposit Requirements', 'Remittance Services', 'Backup Plans & Power Supply', 'Net Banking Features', 'Kotak Card Services & Features', 'Mobile Banking Features', 'Date & Times', 'WhatsApp Banking Services', 'Email Alerts & Preferences', 'Bank Procedures for NACH Compliance', 'Loan Pricing & Terms', 'Loan Information', 'Branch Office Operations & Customer Service', 'Deliverable Timing', 'Dining Locations', 'FCY Cash Withdrawal & Deposit Timings', 'Date & Times', 'Customer Expectations in Banking', 'Cheque Handling Procedures', 'Cheque Rejection Conditions', 'Secure Financial Transactions', 'Bank Rules', 'Bank Account Actions', 'Customer Service Complaints Timeline', 'Financial Security Measures', 'Bank Alerts Tracking', 'Bank Security Measures', 'Secure Internet Banking Access', 'Secure Online Banking Practices', 'Financial Year Rules', 'Bank Customer Services', 'Customer Feedback & Satisfaction', 'Complaint Resolution Instructions', 'Online Grievance Redressal System', 'ATM Complaint Resolution', 'Grievance Filing Instructions', 'Banking Complaints Resolution', 'Bank Services & Commitments', 'Account Branch Access Methods', 'Reserve Bank of India Exchange Facilities', 'Currency Exchanges', 'Bank Note Exchange Facilities', 'Bank Rules for Mutilated Notes', 'Forgery Handling Procedures', 'Reserve Bank of India Security Features', 'Exchange Issues Reporting', 'Customer Service Politeness', 'Damaged Currency Transactions', 'Reserve Bank Branch Services', 'Bank Branch Exchanges', 'Public Feedback', 'Pensioner Services', 'Civil Pension Benefits', 'Age Ranges & Life Expectancy Milestones', 'Pensions Percentages', 'Pension Credit Notifications & Processes', 'RBI Issue Offices Jurisdictions', 'Location Information', 'Location & Contact Details', 'RBI Issue Department Location', 'Contact Details', 'Geographical Locations', 'Telephone Bhavan, Chandigarh Locations', 'Location', 'Address Information', 'Address & Position', 'General Manager Location & Designation', 'Contact Details & Leadership', 'Uttar Pradesh & Uttarakhand Locations', 'Post Bag Location & DGM Responsibility', 'Locations & GMs', 'Contact Information', 'RBI Personnel Locations & Titles', 'Location & Position at RBI Issue Dept', 'Bakery Department Address', 'Confidentiality & Distribution Restrictions', 'Customer Rights Policies', 'Customer Rights & Fairness', 'Fair Treatment Practices', 'Bank Obligations', 'Ethical Practices & Communication Guidelines', 'Communication Policies & Obligations', 'Customer Product Suitability Policies', 'Confidentiality Conditions', 'Customer Information Disclosure Conditions', 'Bank Compensation Policies', 'Grievance Redressal Processes', 'Bank Complaint Handling Timelines', 'Dispute Resolution & Liability for Banks', 'Refund Policy & Bank Inquiries', 'Confidentiality & Usage Restrictions', 'Grievance Redressal Policy & Processes', 'Fair Treatment & Complaint Handling Principles', 'Customer Support Processes', 'Customer Feedback Processes', 'Continuous Improvement for Customer Experience', 'Banking Complaints & Requests', 'Policy Coverage', 'Scope & Applicability', 'Feedback & Complaint Methods', 'Contact Methods & Concern Registration', 'Complaint Escalation Procedures', 'Customer Issue Resolution', 'Complaint Escalation', 'Contact Details', 'Kotak Bank Contact Information', 'Grievance Redressal Process', 'Internal Ombudsman Role & Compliance', 'Internal Ombudsman Roles', 'Standing Committee Composition & Responsibilities', 'Feedback Evaluation', 'Customer Service Compliance Responsibilities', 'Branch-Level CS Committees Purpose & Establishment', 'Customer Service Committees', 'Customer Service Oversight & Policy Development Processes', 'Customer Complaint Escalation in Banking', 'Complaints & Suggestions Arrangements', 'Resolution Times & Turnaround', 'Bank Complaints', 'Date & Times of TAT Measurement', 'Banking Txn Disputes Complaints', 'CRM System Efficiency', 'Complaints Received', 'Customer Complaint Resolution Process', 'Complaint Handling & MIS Processes', 'Customer Feedback & Complaint Handling', 'Customer Satisfaction & Efficiency Strategies', 'Grievance Handling & Complaints', 'Complaint Records RetentionAccessibility', 'Customer Satisfaction Surveys', 'Customer Feedback Improvement Strategies']

        # trying keyword processor as well
        # self.kp = KeywordProcessor()
        # for word in keyword:
        #     self.kp.add_keyword(word)

        # self.keyword_store = Chroma(
        #     collection_name="keyword_index",
        #     chroma_cloud_api_key=os.getenv("CHROMA_API"),
        #     tenant=os.getenv("CHROMA_TENANT"),
        #     database=os.getenv("CHROMA_DB"),
        #     embedding_function = self.embedfn,
        #     collection_metadata={"hnsw:space": "cosine"},
        # )

        self.keyword_store = Chroma(
            collection_name="keyword_index",
            persist_directory = "./chroma_keywords",
            embedding_function = self.embedfn,
            collection_metadata={"hnsw:space": "cosine"},
        )

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # self.vectorstores = {
        #     "podak": Chroma(
        #         collection_name="podak",
        #         chroma_cloud_api_key=os.getenv("CHROMA_API"),
        #         tenant=os.getenv("CHROMA_TENANT"),
        #         database=os.getenv("CHROMA_DB"),
        #         embedding_function = self.embedfn,
        #         collection_metadata={"hnsw:space": "cosine"})
        # }

        self.vectorstores = {
            "podak": Chroma(
                collection_name="podak",
                persist_directory="./podak",
                embedding_function = self.embedfn,
                collection_metadata={"hnsw:space": "cosine"})
        }
        
        print("Number of documents in 'podak' store:", self.vectorstores["podak"]._collection.count())


        self.tools = [self.create_find_context_tool(self.vectorstores), get_geo_tool()]
        self.memory = InMemorySaver()
        self.workflow = self._create_workflow()
        self.config = {"configurable": {"thread_id": "conversation-1"}}

    def create_find_context_tool(self, vectorstores):
        @tool
        def find_context(query: str, collname: str = "podak", no_of_docs: int = 10,
                         keyword: Annotated[List[str], "List of keywords for filtering"] = []) -> str:
            """Searches Kotak Mahindra Bank's official documents for verified answers. 
    
        Parameters:
        - query (str): The exact customer question to search for
        - keyword (str): Extracted keyword(s) from user input for filtering

        Returns:
        - str: Document excerpts or "No results found"
            """

            try:
                collname = 'podak'
                if collname not in vectorstores:
                    return f"Collection '{collname}' not found. Available collections: {list(vectorstores.keys())}"

                # print('Keyword : ', keyword)
                # or_q = [{'$contains' : key} for key in keyword]

                # print('query : ', query)
                # print('keyword in :', keyword)
                    
                
                # matches = vectorstores[collname].similarity_search(query = query, k = no_of_docs, filter = {'keyword' : {'$in' : keyword}}
                matches = vectorstores[collname].similarity_search(query = query, k = no_of_docs
                                                                   )
                if not matches:
                    return f"No relevant documents found in {collname} for query: {query}"
                
                content = "\n\n".join([doc.page_content for doc in matches])
                # print('Content : ', content)
                return f"{content}"
            except Exception as e:
                #logger.error(f"Error in find_context: {e}")
                # print(f"Error in find_context: {e}")
                return f"Error retrieving context: {str(e)}"
        return find_context
    
    def _summarize(self, state: ChatBotState) -> ChatBotState:  #make it async?
        """Running keyword of the chat history."""
        # incomplete for now, unnecessary for short conversations
        keyword = self.get_keyword(state)
        # Try NLP for this
        # keyword = self.kp.extract_keywords(state['user_input'])
        print('keywords :', keyword)
        return {**state, "keyword": keyword}

    def _is_sufficient(self, state: ChatBotState) -> ChatBotState:
        """ Checks if a query is self sufficient. """
        
        query = state['user_input']
        is_waiting = state['is_waiting']

        if is_waiting:
            state['loop_thread'] = state['loop_thread'] + [query]
            return state

        prompt = ChatPromptTemplate.from_template(
            f"""
You are assisting users of **Kotak Mahindra Bank**, so unless the user clearly specifies a different context, assume all questions are about Kotak’s services, policies, or platforms.

---

**Task:**

Given a user input, determine whether it is self-contained and valid.

**Return only the XML-formatted result no explanations, no markdown, no extra text. The XML must be parsable directly by an XML parser. Do not add markdown, do not wrap in code blocks, and do not include ```xml or ``` anywhere. Just give the raw XML content.**

---

**Validation Rules:**

- A valid message must be specific, grammatically complete, and understandable on its own.
- Simple greetings like "hi", "hello", or "good morning" are also valid.
- Strictly return the XML output only and nothing else. No explanation.
- Bank-related questions like "When does the bank open?" or "How do I check my balance?" are valid without additional clarification assume the user means **Kotak Mahindra Bank**.
- **Do not** generate unnecessary follow-up questions for obvious, common banking queries.
- Only ask follow-up questions when absolutely necessary to make sense of the input.
- If the message is extremely grammatically incorrect and cannot be reasonably interpreted, mark it invalid and skip the follow-up questions section.

---

**Output Format (Strictly XML Only):**

If valid:

<response>
  <is_valid>true</is_valid>
</response>

If invalid and clarification is needed:

<response>
  <is_valid>false</is_valid>
  <followup_question>
    <question>...</question>
  </followup_question>
</response>

- If only one follow-up question is needed, include just one `<question>` tag.
- If no meaningful follow-up can be asked, omit the `<followup_question>` section entirely.

---

**Example Valid Inputs:**

- "What is the capital of France?"
- "Hi"
- "Can you help me with internet banking?"
- "When does the bank open?"
- "How do I get my debit card replaced?"
- "Can you find bank branches near Ranchi, Jharkhand ?" - if the city name is given then don't ask a followup question, otherwise ask for a specific location.
- "How to open an account ?" - By account the user means a savings account.

**Example Invalid Inputs:**

- "Why does it work like that?" needs context.
- "Explain this code." missing the code.
- "Is it available?" unclear what 'it' refers to.
- "Branches near me." Ask for users location
- "I have been scammed." Ask the user about what happened.
- "I am scammed." Ask the user about what happened that is the user **hints being scammed**.

---

Now process this user input:

{query}
            """
        )

        
        chain = prompt | self.llm

        result = chain.invoke({
            'query' : query
        })
        
        questions = []
        print(result.content)
        
        try:
            # print(result.content)
            root = ET.fromstring(extract_raw_xml(result.content))
            is_valid = True if root.find("is_valid").text.lower() == 'true' else False


            if is_valid:
                print('Valid question!')
                state['is_valid'] = True
                state['is_waiting'] = False
                state['is_fresh'] = True

                return state

            print('Incomplete question!')

            followup = root.find("followup_question")

            if followup is not None:
                for q in followup.findall("question"):
                    questions.append(q.text)
                
        except Exception as e:
            print(f'Exception in is_valid node : {e}')

        state['messages'] = state['messages'] + [AIMessage(content = questions)]
        loop_thread = state['loop_thread'] + [query]
        return {
            **state,
            'is_valid': is_valid,
            'is_waiting' : True,
            'reference_q' : query,
            'questions' : questions,
            'is_fresh' : True,
            'loop_thread' : loop_thread
            }

    
    def _check_answered(self, state: ChatBotState) -> ChatBotState:
        """Check if followup questions are answered or not"""
        print(state.get('is_fresh', False), state.get('is_waiting', False))
        if state.get('is_fresh', False):
            state['is_fresh'] = False
            return state
        if not state.get('is_waiting', False):
            print('Skipping check answered!')
            return state

        # Check if questions were answered.
        context = state['reference_q']
        query = state['user_input']

        loop_thread = '\n'.join(state['loop_thread'])
        # print(len(loop_thread))
        summary = state.get('reference_sum', 'Nothing')
        # qs = [f'{i + 1}. {q}' for i, q in enumerate(state['questions'])]
        qs = state['questions'][0]  
        # print('checlaaa')
        prompt = ChatPromptTemplate.from_template(f"""
You are a digital assistant for Kotak Mahindra Bank focused on providing clear, helpful responses about banking products and services.

**Input Processing Rules:**

1. FIRST check if the user wants to ask a NEW question (look for phrases like):
   - "Actually, I want to ask about..."
   - "Forget that, how about..."
   - "Instead, can you tell me..."
   - "Let me rephrase..."
   - "New question:..."
   - "Change question to..."

2. If NEW QUESTION detected:
   <response>
   <complete>true</complete>
   <clarified_statement>[EXACT NEW QUESTION USER ASKED]</clarified_statement>
   </response>

3. If USER OPT-OUT detected ("skip", "never mind", etc.):
   <response>
   <complete>false</complete>
   <summary>User changed topic</summary>
   <remaining_questions>
   <question>What would you like to ask instead?</question>
   </remaining_questions>
   </response>

4. Otherwise proceed with NORMAL CLARIFICATION:
   - If all follow-ups answered:
     <response>
     <complete>true</complete>
     <clarified_statement>[combined clear question]</clarified_statement>
     </response>
   - If follow-ups remain:
     <response>
     <complete>false</complete>
     <summary>[progress summary]</summary>
     <remaining_questions>
     <question>[single most important follow-up]</question>
     </remaining_questions>
     </response>

**Current Context:**
Original Question: {context}
Follow-ups Asked: {qs}
User Response: {query}
Previous Summary: {summary}
Previous User responses: {loop_thread}

**Output ONLY the raw XML response:**
        
        """)

        
        print('LLM is confused rn!')
        print(f"Original Question: {context} \n Follow-ups Asked: {qs} \n User Response: {query}\n Previous Summary: {summary} \n Previous User Responses : {loop_thread}")
        chain = prompt | self.llm

        result = chain.invoke({
            'context' : context,
            'query': query,
            'qs': qs
        })

        print('LLM', result.content)
        is_complete = True
        try:
            root = ET.fromstring(extract_raw_xml(result.content))
            is_complete = True if root and root.find('complete').text.lower() == 'true' else False

            if len(state['loop_thread']) > 3:
                state['user_input'] = state['reference_sum'] + ' '  + state['reference_q']
                state['is_complete'] = True

                return {
                    **state,
                    'is_valid': False,
                    'is_waiting' : False,
                    'reference_q' : '',
                    'questions' : [],
                    'loop_thread': []
                }
            elif is_complete:
                followup = root.find("clarified_statement").text
                state['user_input'] = followup
                return {
                    **state,
                    'is_valid': False,
                    'is_waiting' : False,
                    'reference_q' : '',
                    'questions' : [],
                    'loop_thread': []
                }
            else:
                followup = root.find("remaining_questions")
                
                questions = []
                if followup is not None:
                    for q in followup.findall("question"):
                        questions.append(q.text)


                summary = root.find('summary')
                state['reference_sum'] = summary.text
                state['is_waiting'] = True
                # state['questions'] = questions

                state['messages'] = state['messages'] + [AIMessage(content = ' '.join(questions))]

            
        except Exception as e:
            print(f'Exception in check_answered node : {e}')


        return state

        

        
        
    def _react_agent_node(self, state: ChatBotState) -> ChatBotState:
        """Agent for document querying and reasoning."""

        user_input = state["user_input"]
        messages = state.get("messages", [])
        keyword = state["keyword"]
        is_valid = state['is_valid']
        is_waiting = state.get('is_waiting', False)


        if not is_valid:
            if is_waiting:
                return state
            


        # recent conversation context
        context_messages = []
        for msg in messages:
            # passing all messages for now, assuming small conversations
            if isinstance(msg, (HumanMessage, AIMessage)):
                role = "Human" if isinstance(msg, HumanMessage) else "AI"
                context_messages.append(f"{role}: {msg.content}")
        
        conversation_context = "\n".join(context_messages) if context_messages else ""
        
        # system_message = f"""You are a helpful assistant with access to document search tools. 
        # First you are to think about which document to search based on the conversation context and current user input.
        # Then, use the find_context tool to find relevant context based on user query.
        # Finally provide comprehensive answers based on the retrieved context. 
        # Be polite, respectful and accurate. If no relevant information is found, say so clearly.

        # Available documents = {list(self.vectorstores.keys())}

        # Conversation Context: 
        # {conversation_context}

        # Keyword:
        # {keyword}
        
        # Available tools: {[f"{tool.name}: {tool.description}" for tool in self.tools]}
        
        # Think step by step and use tools when needed to answer the user's question."""

#         system_message = f"""You are a helpful digital assistant for Kotak Mahindra Bank products and services. Your purpose is to provide accurate information to customers in a friendly, easy-to-understand manner.


# *Keyword*:{keyword}

# **You MUST follow these steps for EVERY query:**
# 1. **Search First**: Use `find_context` tool with the user's exact query.
# 2. **Analyze**: Check if the tool returned valid Kotak documents.
# 3. **Respond**: Answer ONLY using the tool's output. If none found, say so.



# **Special Cases Handling:**
# - For greetings (hi/hello) or non-banking queries:
#   * DO NOT call tools
#   * Respond with generic welcome message
#   * Example: "Hello! How can I assist you with Kotak banking today?"
# - For Bank timings:
#     * Respond with 9 to 5 time.
#     * Ask the user to contact the local branch office.

        
# **Good Response Example:**
# "To check your account balance, you can use the Kotak Mobile Banking app. Just log in and your balance will show on the dashboard. You can also get mini-statements there."

# **Bad Response Example:**
# "Account balances are visible in mobile banking." 
# (Too vague, didn't verify with search tool)

# **When Information is Unavailable:**
# 1. First state: "Let me check that for you..." (while running search tool)
# 2. If nothing found: "I couldn't find this in current resources. For help, you can:"
#    - Visit kotak.com
#    - Call 1860 266 2666
#    - Message in the mobile app

# **Tone Rules:**
# - Use natural language like "you'll" instead of "you will"
# - Keep responses conversational (2-3 sentences max)
# - Explain terms simply: "FD means Fixed Deposit - like a savings account that earns higher interest"

# **Safety Protocol:**
# If any doubt after searching:
# 1. Admit it: "I want to confirm this for you..."
# 2. Run the search tool again with different keywords
# 3. If still unsure: "For accurate help, please contact customer care at 1860 266 2666"

# **Tool Usage Requirement:**
# - **Every single query** must trigger the search tool first
# - Never answer without running the search
# - Add the user query in the tool.
# - If the tool fails, say: "I'm having trouble accessing that information right now. Please try [alternative option]"

# Example Tool Call:
# find_context(query="...", keyword="['Date & Time']")

# - The keword parameter takes a list of str.

# **Final Reminder:**
# You exist solely to:
# 1. Run the search tool
# 2. Interpret official documents
# 3. Respond conversationally
# 4. Escalate when needed
# """


        system_message = f"""You are a helpful digital assistant for Kotak Mahindra Bank products and services. Your purpose is to provide accurate information to customers in a friendly, easy-to-understand manner.

        **REQUIREMENT** is that you are the backend of a voice chat bot. Hence generate responses as concisely and accurately as possible.

*Keyword*:{keyword}
        
**ALWAYS FOLLOW THIS DECISION FLOW:**
 Step 1: Classify the query type

    
    If the user's input includes any of the following types of phrases:
        - "near me", "nearest", "around me"
        - "in [Location]", "at [Place]", "from [Location]"
        - Names of cities, towns, or regions in India (e.g. Mumbai, Ranchi, Guwahati, Andheri, etc.)

        Then:
        1. Extract the **location** from the query.
        2. Call this tool: `geo_tool(location=extracted_location)`

        Examples:
        - "Bank branch in Ranchi" -> geo_tool(location="Ranchi")
        - "Bank branch in Ranchi, Jharkhand" -> geo_tool(location="Ranchi, Jharkhand")
        - "Bank branch in Ranchi, Jharkhand, India" -> geo_tool(location="Ranchi, Jharkhand, India")
        - "Bank branch in [LOCATION]" -> geo_tool(location="[LOCATION]")
        - "Kotak ATM near me" -> geo_tool(location="user's current location")
        - "ATM in Guwahati" -> geo_tool(location="Guwahati")
        - "check for kotak bank branch in kolkata" -> geo_tool(location="Kolkata")
        - "check for kotak bank branch in [LOCATION]" -> geo_tool(location="[LOCATION]")


    Otherwise:
        Call: find_context(query="{user_input}", keyword={keyword})

**Additional Rules:**
        1. Greetings or Non-Banking Messages
        DO NOT call any tools
        Reply: "Hello! How can I assist you with Kotak banking today?"

        2. Bank Timings or Lunch Timings
        DO NOT call tools
        For bank timings in general reply
        Reply: "Kotak branch timings are generally from 9 AM to 5 PM. For holidays or local variations, please contact your branch."
        For lunch timings reply
        Reply: "Lunch timings are between 1pm to 2pm in Kotak Bank."

        3. Geo Tool Failure or No Location
        If the location isn't available or geo_tool fails, respond:
        "Could you please share your location so I can find the nearest Kotak branch or ATM for you?"

**Response Format:**

    Use ONLY the tool output (never guess)
    Keep answers conversational and concise (23 sentences)
    Explain simply (e.g. "FD means Fixed Deposit like a savings account that earns higher interest")
    If no info found, respond:
        "Let me check that for you..."

        "I couldn't find this in current resources. For help, you can:"  Visit kotak.com Call 1860 266 2666 Message in the mobile app"

**Safety Protocol:**

        If you're ever unsure:
        Admit it: "I want to confirm this for you..."
        Retry with broader terms (if applicable)
        Or say: "For accurate help, please contact Kotak Customer Care at 1860 266 2666"


**Good Response Example:**
"To check your account balance, you can use the Kotak Mobile Banking app. Just log in and your balance will show on the dashboard. You can also get mini-statements there."

**Bad Response Example:**
"Account balances are visible in mobile banking." 
(Too vague, didn't verify with search tool)
        """

        print('Reached end node!')

        agent_graph = create_langgraph_react_agent(
            self.llm, 
            self.tools, 
            prompt=system_message
        )
        
        try:
            
            agent_state = {"messages": [HumanMessage(content=user_input)]}
            result = agent_graph.invoke(agent_state)
            
            final_message = result["messages"][-1]
            response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            new_messages = [
                HumanMessage(content=user_input),
                AIMessage(content=response_content)
            ]
            
            existing_messages = state.get("messages", [])
            updated_messages = existing_messages + new_messages
            
            return {
                **state,
                "messages": updated_messages
            }
            
        except Exception as e:
            print(f"Encountered Exception in react agent node: {e}")
            error_response = f"I encountered an error while processing your request: {str(e)}"
            return {
                **state,
                "messages": [
                    HumanMessage(content=user_input),
                    AIMessage(content=error_response)
                ]
            }

    def _create_workflow(self) -> StateGraph:
        """Creating Langgraph workflow"""

        workflow = StateGraph(ChatBotState)
        
        workflow.add_node("summarize_messages", self._summarize)
        workflow.add_node("react_agent", self._react_agent_node)
        workflow.add_node('is_sufficient', self._is_sufficient)
        workflow.add_node('check_answered', self._check_answered)
        # workflow.add_node("conversation_history", self.get_conversation_history)
        
        workflow.add_edge(START, "summarize_messages")
        workflow.add_edge("summarize_messages", "is_sufficient")
        workflow.add_edge('is_sufficient', 'check_answered')
        workflow.add_edge('check_answered', 'react_agent')
        # workflow.add_edge("react_agent", "conversation_history")
        # workflow.add_edge("conversation_history", END)
        workflow.add_edge("react_agent", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def chat(self, user_input: str) -> str:
        """Main chat function using LangGraph memory management"""
        try:
            # getting existing state from memory or create initial state
            try:
                existing_state = self.workflow.get_state(self.config)
                if existing_state and existing_state.values:
                    # continue existing conversation
                    curr_state = {
                        "messages": existing_state.values.get("messages", []),
                        "user_input": user_input,            
                        "keyword": existing_state.values.get("keyword", ""),
                        'is_valid' : existing_state.values.get('is_valid', False),
                        'is_waiting' : existing_state.values.get('is_waiting', False),
                        'questions' : existing_state.values.get('questions', []),
                        'reference_q': existing_state.values.get('reference_q', []),
                        'reference_sum': existing_state.values.get('reference_sum', ""),
                        'loop_thread' : existing_state.values.get('loop_thread', [])
                    }
                else:
                    # start new conversation
                    curr_state = {
                        "messages": [],
                        "user_input": user_input,                        
                        "keyword": "",
                        'is_valid' : False,
                        'is_waiting' : False,
                        'questions' : [],
                        'reference_q' : '',
                        'reference_sum': '',
                        'loop_thread': []
                    }
            except Exception as e:
                print(f"Could not retrieve existing state: {e}, starting fresh")
                # fallback to new conversation
                curr_state = {
                        "messages": [],
                        "user_input": user_input,                        
                        "keyword": "",
                        'is_valid' : False,
                        'is_waiting' : False,
                        'questions' : [],
                        'reference_q' : '',
                        'reference_sum': '',
                        'loop_thread': []
                }

            # Do asynchronously
            # result = await self.workflow.ainvoke(curr_state, self.config)
            result = self.workflow.invoke(curr_state, self.config)
            
            messages = result.get("messages", [])
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm sorry, I couldn't process your request properly."
                
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."

    async def achat(self, user_input: str) -> str:
        """Async Main chat function using LangGraph memory management"""
        try:
            # getting existing state from memory or create initial state
            try:
                existing_state = self.workflow.get_state(self.config)
                if existing_state and existing_state.values:
                    # continue existing conversation
                    curr_state = {
                        "messages": existing_state.values.get("messages", []),
                        "user_input": user_input,            
                        "keyword": existing_state.values.get("keyword", ""),
                        'is_valid' : existing_state.values.get('is_valid', False),
                        'is_waiting' : existing_state.values.get('is_waiting', False),
                        'questions' : existing_state.values.get('questions', []),
                        'reference_q': existing_state.values.get('reference_q', []),
                        'reference_sum': existing_state.values.get('reference_sum', ""),
                        'loop_thread' : existing_state.values.get('loop_thread', [])
                    }
                else:
                    # start new conversation
                    curr_state = {
                        "messages": [],
                        "user_input": user_input,                        
                        "keyword": "",
                        'is_valid' : False,
                        'is_waiting' : False,
                        'questions' : [],
                        'reference_q' : '',
                        'reference_sum': '',
                        'loop_thread' : []
                    }
            except Exception as e:
                print(f"Could not retrieve existing state: {e}, starting fresh")
                # fallback to new conversation
                curr_state = {
                        "messages": [],
                        "user_input": user_input,                        
                        "keyword": "",
                        'is_valid' : False,
                        'is_waiting' : False,
                        'questions' : [],
                        'reference_q' : '',
                        'reference_sum': '',
                        'loop_thread' : []
                }

            # Do asynchronously
            # result = await self.workflow.ainvoke(curr_state, self.config)
            result = await self.workflow.ainvoke(curr_state, self.config)
            
            messages = result.get("messages", [])
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm sorry, I couldn't process your request properly."
                
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try again."
        
    def get_full_conversation_history(self):
        """Get the current conversation history"""
        try:
            existing_state = self.workflow.get_state(self.config)
            if existing_state and existing_state.values:
                messages = existing_state.values.get("messages", [])
                history = []
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        role = "Human" if isinstance(msg, HumanMessage) else "AI"
                        history.append(f"{role}: {msg.content}")
                return history
            else:
                print("No conversation history found.")
                return([])
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return([])
        
    def get_keyword(self, state: ChatBotState) -> str:
        """Get updated keyword"""
        query = state['user_input']
        threshold = 0.5

        # keyword_prompt = ChatPromptTemplate.from_template(
        #     f"""
        #     **Task**: Extract the most relevant keyword(s) from the predefined list below that matches the user's query. 
        #     Return ONLY the exact keyword(s) or say "No match found".

        #     **Rules**:
        #     1. Strictly use ONLY the keywords from this list: {keyword}
        #     2. Ignore partial matches or synonyms. 
        #     3. Return maximum 5 keywords if multiple are relevant.

        #     **Query**: {query}

        #     **Output Format** (comma-separated or "No match found"):
        #     """
        # )


        # keyword_chain = keyword_prompt | self.llm

        # result = keyword_chain.invoke({
        #     "keywords": ", ".join(keyword),
        #     "query": query
        # })

        # # print(result)
        # # print('test : ', list(map(lambda x : x.strip(), result.content.split(','))))

        # return list(map(lambda x : x.strip(), result.content.split(',')))

        try:
            matches = self.keyword_store.similarity_search_with_relevance_scores(query, k=5)

            filtered = [doc.page_content for doc, score in matches if score >= threshold]

            return filtered if filtered else ["No match found"]
        except Exception as e:
            print(f"Error in Chroma keyword matching: {e}")
            return ["No match found"]
        
    def clear_memory(self):
        """Clear the conversation memory"""
        try:
            # new thread ID 
            import uuid
            self.config = {"configurable": {"thread_id": f"conversation-{uuid.uuid4()}"}}
            print("Memory cleared - started new conversation thread")
        except Exception as e:
            print(f"Error clearing memory: {e}")


if __name__ == "__main__":

    agent = MyHelpfulBot()
    #print(agent.find_context.args_schema.model_json_schema()) # can add Annotated args 
    print("Chatbot initialised! Type 'quit' to exit, 'clear' to clear memory.")
    print("You can ask questions about the documents in your collection.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                #print(agent.get_keyword())
                print(agent.get_full_conversation_history())
                break
            elif user_input.lower() == 'clear':
                agent.clear_memory()
                print("Memory cleared!")
                continue
            elif not user_input:
                continue
            
            # Get response from the bot
            start = time.time()
            response = agent.chat(user_input)
            time_taken = time.time()-start
            print(f"\nBot: {response}\n({time_taken} seconds)")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

    #print("Conversation history: \n\n")
    #print(agent.get_full_conversation_history())

'''
#testing the tool

if __name__ == "__main__":
    agent = MyHelpfulBot()
    
    find_context_tool = agent.create_find_context_tool(agent.vectorstores)
    
    # Test 1: Valid collection and query
    print("=== Test 1: Valid query ===")
    result = find_context_tool.invoke({
        "query": "prisoner", 
        "collname": "Prison", 
        "no_of_docs": 3
    })
    print(f"Result: {result}")
    print(f"Length: {len(result)} characters")
    
    # Test 2: Invalid collection
    print("\n=== Test 2: Invalid collection ===")
    result = find_context_tool.invoke({
        "query": "test", 
        "collname": "NonExistent", 
        "no_of_docs": 3
    })
    print(f"Result: {result}")
    
    # Test 3: Different collection
    print("\n=== Test 3: Different collection ===")
    result = find_context_tool.invoke({
        "query": "horror", 
        "collname": "Seismic__horror", 
        "no_of_docs": 2
    })
    print(f"Result: {result}")


agent = MyHelpfulBot(model="qwen2.5:3b")
print("init!")
print(agent.workflow.get_state(agent.config))
agent.clear_memory()
print(agent.workflow.get_state(agent.config))
print(agent.config["configurable"]["thread_id"])'''
