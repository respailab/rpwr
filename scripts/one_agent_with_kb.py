import os
import time
import argparse
import pandas as pd
from typing import TypedDict
from dotenv import load_dotenv
from tqdm import tqdm
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama


# ------------------- Load Environment Variables -------------------
load_dotenv()

# ------------------- Prompt Template -------------------
DIAGNOSIS_PROMPT_TEMPLATE = """
Historical samples of diagnosis: {historical_data}. \n\n
Patient information collected by survey: {patient_symptoms}\n\n
Compare the historical data and given patient symptoms and provide final diagnosis as 'Rheumatoid Arthritis' or 'Not Rheumatoid Arthritis'.\n\n
Do not output anything else.
"""

REASONING_PROMPT_TEMPLATE = """
Patient information collected by survey: {patient_symptoms}\n\n
Diagnosis - {diagnosis}
Provide reasons of diagnosis based on the following factors:
    1. Presence of early morning stiffness: higher is the duration of early morning stiffness, more is the chance of having an inflammatory arthritis 
    2. Pain worsens after rest and improves with activity
    3. Involvement of the wrists, and small joints of the hands or feet makes it more likely however other peripheral joints may also be involved
    4. Good response to painkillers 
    5. Additive distribution 
    6. Gradual evolution of deformities 
    7. Previous history of pain and swelling in specific joints as opposed to widespread swelling of body parts or swelling in all joints 
    8. Absence of axial involvement or mid-foot involvement, especially in the first few years of the disease 
    9. Definitive swelling in specific joints as opposed to widespread swelling of body parts or swelling in all joints 
    10. Presence of skin rash is not seen in Rheumatoid arthritis and suggestive of other connective tissue disorders
    11. Low grade fever may be associated with Rheumatoid arthritis, but high-grade fever is unlikely
    12. Dryness of eyes and Dryness of mouth can be associated with Rheumatoid Arthritis


Mention reasons in support of this diagnosis. 
Mention points against this diagnosis. 
Provide explanation for considering the diagnosis of Rheumatoid Arthritis and Not Rheumatoid Arthritis
Also mention the other possible differential for that patient 
Do not output anything else.
"""

# ------------------- Define Custom LLM State -------------------
class LLMState(TypedDict):
    prompt: str
    diagnosis: str
    reasons: str

# ------------------- Patient Data Parsing Function -------------------
def parse_patient_data(raw_data_list):
    """
    Parses raw patient data into structured dictionaries.

    Args:
        raw_data_list (list): List of raw patient data strings.

    Returns:
        List[dict]: Parsed patient information with doctor's diagnosis.
    """
    parsed_results = []

    for entry in raw_data_list:
        entry = entry.strip()
        if not entry:
            continue

        parts = entry.split("Final Diagnosis:")
        if len(parts) != 2:
            print(f"Warning: Unexpected data format:\n{entry}")
            continue

        patient_info_block = parts[0].strip()
        final_diagnosis = parts[1].strip()

        patient_info = {}
        for line in patient_info_block.split('\n')[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                patient_info[key.strip()] = value.strip()

        parsed_results.append({
            "patient_info": patient_info,
            "doctors_diagnosis": final_diagnosis
        })

    return parsed_results

# ------------------- Save Inference Results to CSV -------------------
def save_results_to_csv(results, output_dir, filename):
    """
    Saves inference results to a CSV file.

    Args:
        results (list): [reasons, predicted_labels, actual_labels]
        output_dir (str): Output directory path
        filename (str): Output file name (without extension)
    """
    df = pd.DataFrame({
        "Reasons": results[0],
        "Predicted Labels": results[1],
        "Actual Labels": results[2]
    })

    try:
        full_path = os.path.join(output_dir, f"{filename}.csv")
        df.to_csv(full_path, index=False)
    except:
        df.to_csv("unsaved.csv", index=False)
    print(f"Results saved to {full_path}")

# ------------------- LLM Diagnostic Agent -------------------
def run_diagnosis_agent(state: LLMState, retriever, llm, wait_time: int):
    """
    Passes the patient prompt through the LLM to get a diagnosis.

    Args:
        state (LLMState): Input state with the prompt.
        retriever: Knowledge base retriever
        llm: LLM instance (Google or OpenAI).
        wait_time (int): Time delay (Google rate limiting workaround).

    Returns:
        dict: Output state with diagnosis.
    """
    docs = retriever.invoke(state['prompt'])
    docs_text = ["".join(d.page_content) for d in docs]
    historical_json = parse_patient_data(docs_text)
    prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(historical_data=historical_json, patient_symptoms=state['prompt'])
    response = llm.invoke(prompt)
    time.sleep(wait_time)
    return {"diagnosis": response.content}

# ------------------- LLM Reasoning Agent -------------------
def run_reasoning_agent(state: LLMState, llm, wait_time: int):
    """
    Executes the reasoning agent to generate explanations for a given diagnosis.

    Args:
        state (LLMState): A dictionary containing the following keys:
            - 'prompt': The patient's symptoms and information as a string.
            - 'diagnosis': The diagnosis generated by the diagnostic agent.
        llm: The language model instance (e.g., Google Generative AI or OpenAI).
        wait_time (int): Time delay in seconds to handle rate-limiting for the LLM.

    Returns:
        dict: A dictionary containing the following key:
            - 'reasons': The reasoning output generated by the LLM as a string.
    """
    prompt = REASONING_PROMPT_TEMPLATE.format(patient_symptoms=state['prompt'], diagnosis=state['diagnosis'])
    response = llm.invoke(prompt)
    time.sleep(wait_time)
    return {"reasons": response.content}


# ------------------- Main Pipeline -------------------
def main():
    # ------------------- Argument Parser -------------------
    parser = argparse.ArgumentParser(description="Diagnosis agent (no knowledge base).")
    parser.add_argument("--provider", type=str, required=True, choices=["GOOGLE", "OPENAI", "OLLAMA"],
                        help="LLM provider: GOOGLE or OPENAI")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gemini-2.5-pro-preview-03-25", "gemini-2.0-flash", "o1", "o3-mini", "qwq", "deepseek-r1:70b"],
                        help="Model name")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory to save results")
    args = parser.parse_args()

    print(f"Arguments: {args}")
    search_args = 5

    # ------------------- Initialize LLM and Knowledge Base-------------------
    if args.provider == "GOOGLE":
        wait_time = 13  # Prevents rate limit issues
        llm = ChatGoogleGenerativeAI(model=args.model, api_key=os.getenv("GOOGLE_API_KEY"))
    elif args.provider == "OPENAI":
        wait_time = 0
        llm = ChatOpenAI(model=args.model, api_key=os.getenv("OPENAI_API_KEY"))
    elif args.provider == "OLLAMA":
        search_args = 2
        wait_time = 0
        llm = ChatOllama(model=args.model, temperature=0.8)
        
    print(f"Using model: {args.model}")

    retriever = Chroma(embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY)")), persist_directory="knowledge_base").as_retriever(search_kwargs={"k": search_args})

    # ------------------- Load Test Dataset -------------------
    test_df = pd.read_csv("data/test_data.csv")
    prompts = test_df["Patient Prompt"].tolist()
    actual_labels = test_df["Actual Diagnosis"].tolist()

    # ------------------- Create Output Directory -------------------
    os.makedirs(args.results_dir, exist_ok=True)

    # ------------------- Define Workflow Graph -------------------
    workflow = StateGraph(LLMState)
    workflow.add_node("Diagnosis", lambda state: run_diagnosis_agent(state, retriever, llm, wait_time))
    workflow.add_node("Reasoning", lambda state: run_reasoning_agent(state, llm, wait_time))
    workflow.add_edge(START, "Diagnosis")
    workflow.add_edge("Diagnosis", "Reasoning")
    workflow.add_edge("Reasoning", END)
    diagnostic_chain = workflow.compile()

    # ------------------- Run Inference -------------------
    predicted_labels = []
    reasoning_outputs = []

    for idx, patient_prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Diagnosing Patients"):
        state = diagnostic_chain.invoke({"prompt": patient_prompt})
        if args.provider == "OLLAMA":
            try:
                predicted_labels.append(state['diagnosis'].split("</think>")[1])
                reasoning_outputs.append(state['reasons'].split("</think>")[1])  # Placeholder for reasoning if needed
            except:
                predicted_labels.append("NA")
                reasoning_outputs.append("NA")
        else:
            predicted_labels.append(state['diagnosis'])
            reasoning_outputs.append(state['reasons']) 

        # Debugging: Only run once if needed for quick testing
        # break

        # ------------------- Save Results -------------------
        # result_bundle = [reasoning_outputs, predicted_labels, actual_labels[0]]
        print(idx)
        t = idx + 1
        result_bundle = [reasoning_outputs[:t], predicted_labels[:t], actual_labels[:t]]
        save_results_to_csv(result_bundle, args.results_dir, args.model)

    result_bundle = [reasoning_outputs, predicted_labels, actual_labels]
    save_results_to_csv(result_bundle, args.results_dir, args.model)

    print("Results saved successfully!")


# ------------------- Entry Point -------------------
if __name__ == "__main__":
    main()
